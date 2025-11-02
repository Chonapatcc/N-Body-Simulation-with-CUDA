#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        cudaError_t err__ = (expr);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)          \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

constexpr float G_CONST = 6.67430e-11f;
constexpr float EPSILON_CONST = 1e-3f;

struct Particle {
    float x, y, z;    // Position
    float vx, vy, vz; // Velocity
    float mass;
};

int loadParticlesFromCSV(std::vector<Particle>& particles, const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return -1;
    }

    std::string line;
    bool is_header = true;
    int n_loaded = 0;

    while (std::getline(file, line)) {
        if (is_header) {
            is_header = false;
            continue;
        }
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string cell;
        std::vector<float> values;

        while (std::getline(ss, cell, ',')) {
            try {
                values.push_back(std::stof(cell));
            } catch (const std::exception&) {
                std::cerr << "Warning: Bad data in " << filename << " on line: " << line << std::endl;
                values.clear();
                break;
            }
        }

        if (values.size() >= 7) {
            Particle p{};
            p.x = values[0];
            p.y = values[1];
            p.z = values[2];
            p.vx = values[3];
            p.vy = values[4];
            p.vz = values[5];
            p.mass = values[6];
            particles.push_back(p);
            n_loaded++;
        } else if (!values.empty()) {
            std::cerr << "Warning: Skipping malformed line in " << filename
                      << " (expected >= 7 columns): " << line << std::endl;
        }
    }

    file.close();
    std::cout << "Successfully loaded " << n_loaded << " particles from " << filename << std::endl;
    return n_loaded;
}

std::vector<std::string> getCSVFiles(const std::string& directoryPath) {
    std::vector<std::string> csvFiles;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".csv") {
                csvFiles.push_back(entry.path().string());
            }
        }
    } catch (std::filesystem::filesystem_error& e) {
        std::cerr << "Error accessing directory " << directoryPath << ": " << e.what() << std::endl;
    }
    return csvFiles;
}

void appendTimingForFile(const std::string& filename, long long durationMs) {
    const std::filesystem::path inputPath(filename);
    const std::filesystem::path timingDir = inputPath.parent_path() / "timelog";
    try {
        std::filesystem::create_directories(timingDir);
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error creating timing directory " << timingDir << ": " << e.what() << std::endl;
        return;
    }

    const std::filesystem::path perFilePath = timingDir / inputPath.filename();
    const bool isNewFile = !std::filesystem::exists(perFilePath);

    std::ofstream out(perFilePath, std::ios::app);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open timing file " << perFilePath << std::endl;
        return;
    }

    if (isNewFile) {
        out << "duration_ms\n";
    }
    out << durationMs << '\n';
}

void writeTimingSummary(const std::string& directoryPath, const std::vector<std::pair<std::string, long long>>& timings) {
    if (timings.empty()) {
        return;
    }

    std::filesystem::path root(directoryPath);
    std::filesystem::path targetDir = root;
    if (!root.has_filename()) {
        targetDir = root.parent_path();
    }

    std::filesystem::path timingDir = targetDir / "timelog";
    try {
        std::filesystem::create_directories(timingDir);
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error creating timing directory " << timingDir << ": " << e.what() << std::endl;
        return;
    }

    std::string summaryName = targetDir.filename().string();
    if (summaryName.empty()) {
        summaryName = "timelog";
    }

    std::filesystem::path summaryPath = timingDir / (summaryName + ".csv");
    std::ofstream out(summaryPath);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open summary timing file " << summaryPath << std::endl;
        return;
    }

    out << "filename,duration_ms\n";
    for (const auto& entry : timings) {
        out << std::filesystem::path(entry.first).filename().string() << "," << entry.second << '\n';
    }
}

__global__ void nbodyUltimateKernel(const float4* __restrict__ posMass_in,
                                  const float4* __restrict__ vel_in,
                                  float4* __restrict__ posMass_out,
                                  float4* __restrict__ vel_out,
                                  int N,
                                  float dt) {
    extern __shared__ float4 tilePosMass[];

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    if (gid >= N) {
        return;
    }

    float4 selfPosMass = posMass_in[gid];
    float x = selfPosMass.x;
    float y = selfPosMass.y;
    float z = selfPosMass.z;
    float mass = selfPosMass.w;

    float4 selfVel = vel_in[gid];
    float vx = selfVel.x;
    float vy = selfVel.y;
    float vz = selfVel.z;

    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;

    for (int tileBase = 0; tileBase < N; tileBase += blockDim.x) {
        const int jGlobal = tileBase + tid;
        if (jGlobal < N) {
            tilePosMass[tid] = posMass_in[jGlobal];
        }
        __syncthreads();

        const int tileCount = min(blockDim.x, N - tileBase);
        
        #pragma unroll 8
        for (int j = 0; j < tileCount; ++j) {
            const int neighborIdx = tileBase + j;

            if (neighborIdx == gid) {
                continue;
            }

            const float4 other = tilePosMass[j];

            const float dx = other.x - x;
            const float dy = other.y - y;
            const float dz = other.z - z;

            const float dist_sq = dx * dx + dy * dy + dz * dz + EPSILON_CONST;
            const float inv_dist = rsqrtf(dist_sq);
            const float inv_dist3 = inv_dist * inv_dist * inv_dist;

            const float scalar = G_CONST * other.w * inv_dist3;

            ax = fmaf(scalar, dx, ax);
            ay = fmaf(scalar, dy, ay);
            az = fmaf(scalar, dz, az);
        }

        __syncthreads();
    }

    vx = fmaf(ax, dt, vx);
    vy = fmaf(ay, dt, vy);
    vz = fmaf(az, dt, vz);

    x = fmaf(vx, dt, x);
    y = fmaf(vy, dt, y);
    z = fmaf(vz, dt, z);

    vel_out[gid] = make_float4(vx, vy, vz, 0.0f);
    posMass_out[gid] = make_float4(x, y, z, mass);
}


int main() {
    const float dt = 0.01f;
    const int STEPS = 10;
    const std::string directoryPath = "data/star_cluster_simulation";

    std::vector<std::string> csvFiles = getCSVFiles(directoryPath);
    if (csvFiles.empty()) {
        std::cerr << "No .csv files found in " << directoryPath << std::endl;
        return 1;
    }

    std::cout << "Found " << csvFiles.size() << " files to process." << std::endl;
    std::cout << "========================================" << std::endl;

    std::vector<std::pair<std::string, long long>> timingSummary;

    CUDA_CHECK(cudaFuncSetAttribute(
        nbodyUltimateKernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100));

    for (const std::string& filename : csvFiles) {
        std::vector<Particle> particles;
        int N = loadParticlesFromCSV(particles, filename);

        if (N <= 0) {
            std::cerr << "Failed to load " << filename << " or file is empty. Skipping." << std::endl;
            std::cout << "----------------------------------------" << std::endl;
            continue;
        }

        int minGridSize = 0;
        int blockSize = 0;
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
            &minGridSize,
            &blockSize,
            nbodyUltimateKernel,
            0,
            0));
        
        blockSize = std::clamp(blockSize, 128, 512);
        
        const int blocks = (N + blockSize - 1) / blockSize;
        const size_t sharedBytes = static_cast<size_t>(blockSize) * sizeof(float4);

        std::vector<float4> h_posMass(N);
        std::vector<float4> h_vel(N);
        for (int i = 0; i < N; ++i) {
            h_posMass[i] = make_float4(particles[i].x,
                                       particles[i].y,
                                       particles[i].z,
                                       particles[i].mass);
            h_vel[i] = make_float4(particles[i].vx,
                                   particles[i].vy,
                                   particles[i].vz,
                                   0.0f);
        }

        const size_t bytesVec4 = static_cast<size_t>(N) * sizeof(float4);

        CUDA_CHECK(cudaHostRegister(h_posMass.data(), bytesVec4, cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister(h_vel.data(), bytesVec4, cudaHostRegisterDefault));

        float4* d_posMass[2];
        float4* d_vel[2];
        for (int idx = 0; idx < 2; ++idx) {
            CUDA_CHECK(cudaMalloc(&d_posMass[idx], bytesVec4));
            CUDA_CHECK(cudaMalloc(&d_vel[idx], bytesVec4));
        }

        cudaStream_t copyStream{};
        cudaStream_t computeStream{};
        CUDA_CHECK(cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));

        cudaEvent_t startEvent{};
        cudaEvent_t stopEvent{};
        cudaEvent_t h2dReady{};
        CUDA_CHECK(cudaEventCreate(&startEvent));
        CUDA_CHECK(cudaEventCreate(&stopEvent));
        CUDA_CHECK(cudaEventCreateWithFlags(&h2dReady, cudaEventDisableTiming));

        CUDA_CHECK(cudaEventRecord(startEvent, copyStream));
        
        CUDA_CHECK(cudaMemcpyAsync(d_posMass[0], h_posMass.data(), bytesVec4, cudaMemcpyHostToDevice, copyStream));
        CUDA_CHECK(cudaMemcpyAsync(d_vel[0], h_vel.data(), bytesVec4, cudaMemcpyHostToDevice, copyStream));
        
        CUDA_CHECK(cudaEventRecord(h2dReady, copyStream));
        
        CUDA_CHECK(cudaStreamWaitEvent(computeStream, h2dReady, 0));


        std::cout << "Starting GPU N-Body simulation (Ultimate: streams + tuned kernel)..." << std::endl;
        std::cout << "Using optimal blockSize: " << blockSize << " (GridSize: " << blocks << ")" << std::endl;

        int current = 0;
        int next = 1;

        for (int step = 0; step < STEPS; ++step) {
            nbodyUltimateKernel<<<blocks, blockSize, sharedBytes, computeStream>>>(
                d_posMass[current],
                d_vel[current],
                d_posMass[next],
                d_vel[next],
                N,
                dt);
            CUDA_CHECK(cudaPeekAtLastError());

            if ((step + 1) % std::max(1, STEPS / 10) == 0) {
                float percent = (step + 1) * 100.0f / static_cast<float>(STEPS);
                std::cout << "\rProgress: " << std::fixed << std::setprecision(1)
                          << percent << "%" << std::flush;
            }

            std::swap(current, next);
        }
        std::cout << std::endl;

        CUDA_CHECK(cudaEventRecord(stopEvent, computeStream));
        
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        CUDA_CHECK(cudaStreamSynchronize(copyStream));

        float elapsedMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
        long long durationMs = static_cast<long long>(std::llround(elapsedMs));

        timingSummary.emplace_back(filename, durationMs);
        appendTimingForFile(filename, durationMs);

        std::cout << "GPU simulation for " << filename << " finished in "
                  << durationMs << " ms." << std::endl;
        std::cout << "========================================" << std::endl;

        CUDA_CHECK(cudaEventDestroy(h2dReady));
        CUDA_CHECK(cudaEventDestroy(stopEvent));
        CUDA_CHECK(cudaEventDestroy(startEvent));
        CUDA_CHECK(cudaStreamDestroy(computeStream));
        CUDA_CHECK(cudaStreamDestroy(copyStream));

        for (int idx = 0; idx < 2; ++idx) {
            CUDA_CHECK(cudaFree(d_posMass[idx]));
            CUDA_CHECK(cudaFree(d_vel[idx]));
        }

        CUDA_CHECK(cudaHostUnregister(h_posMass.data()));
        CUDA_CHECK(cudaHostUnregister(h_vel.data()));
    }

    writeTimingSummary(directoryPath, timingSummary);

    std::cout << "All GPU simulations complete." << std::endl;
    return 0;
}
