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

#define CUDA_CHECK(expr)                                                         \
    do {                                                                         \
        cudaError_t err__ = (expr);                                              \
        if (err__ != cudaSuccess) {                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

constexpr float G_CONST = 6.67430e-11f;
constexpr float EPSILON_CONST = 1e-3f;

struct Particle {
    float x, y, z;
    float vx, vy, vz;
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

__global__ void nbodyCoalescedKernel(const float4* posMass_in,
                                     const float4* vel_in,
                                     float4* posMass_out,
                                     float4* vel_out,
                                     int N,
                                     float dt) {
    extern __shared__ float4 sharedPosMass[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    float4 selfPosMass = posMass_in[i];
    float x = selfPosMass.x;
    float y = selfPosMass.y;
    float z = selfPosMass.z;
    float mass = selfPosMass.w;

    float4 selfVel = vel_in[i];
    float vx = selfVel.x;
    float vy = selfVel.y;
    float vz = selfVel.z;

    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;

    for (int tile = 0; tile < N; tile += blockDim.x) {
        int jGlobal = tile + threadIdx.x;
        sharedPosMass[threadIdx.x] = jGlobal < N ? posMass_in[jGlobal]
                                                 : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        __syncthreads();

        int tileCount = min(blockDim.x, N - tile);
        for (int j = 0; j < tileCount; ++j) {
            int neighborIndex = tile + j;
            if (neighborIndex == i) {
                continue;
            }

            float4 other = sharedPosMass[j];
            float dx = other.x - x;
            float dy = other.y - y;
            float dz = other.z - z;
            float dist_sq = dx * dx + dy * dy + dz * dz + EPSILON_CONST;
            float dist = sqrtf(dist_sq);
            float inv_dist_cubed = 1.0f / (dist_sq * dist);
            float force_scalar = G_CONST * mass * other.w * inv_dist_cubed;
            ax += force_scalar * dx / mass;
            ay += force_scalar * dy / mass;
            az += force_scalar * dz / mass;
        }
        __syncthreads();
    }

    vx += ax * dt;
    vy += ay * dt;
    vz += az * dt;

    x += vx * dt;
    y += vy * dt;
    z += vz * dt;

    vel_out[i] = make_float4(vx, vy, vz, 0.0f);
    posMass_out[i] = make_float4(x, y, z, mass);
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

    for (const std::string& filename : csvFiles) {
        std::vector<Particle> particles;
        int N = loadParticlesFromCSV(particles, filename);

        if (N <= 0) {
            std::cerr << "Failed to load " << filename << " or file is empty. Skipping." << std::endl;
            std::cout << "----------------------------------------" << std::endl;
            continue;
        }

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

        float4* d_posMass[2];
        float4* d_vel[2];

        for (int idx = 0; idx < 2; ++idx) {
            CUDA_CHECK(cudaMalloc(&d_posMass[idx], bytesVec4));
            CUDA_CHECK(cudaMalloc(&d_vel[idx], bytesVec4));
        }

        CUDA_CHECK(cudaMemcpy(d_posMass[0], h_posMass.data(), bytesVec4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vel[0], h_vel.data(), bytesVec4, cudaMemcpyHostToDevice));

        std::cout << "Starting GPU N-Body simulation (coalesced global memory)..." << std::endl;

        cudaEvent_t startEvent{};
        cudaEvent_t stopEvent{};
        CUDA_CHECK(cudaEventCreate(&startEvent));
        CUDA_CHECK(cudaEventCreate(&stopEvent));
        CUDA_CHECK(cudaEventRecord(startEvent));

        int current = 0;
        int next = 1;

        const int threadsPerBlock = 256;
        const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        const size_t sharedBytes = static_cast<size_t>(threadsPerBlock) * sizeof(float4);

        for (int step = 0; step < STEPS; ++step) {
            nbodyCoalescedKernel<<<blocks, threadsPerBlock, sharedBytes>>>(
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
        CUDA_CHECK(cudaEventRecord(stopEvent));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));
        std::cout << std::endl;

        float elapsedMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
        long long durationMs = static_cast<long long>(std::llround(elapsedMs));

        CUDA_CHECK(cudaEventDestroy(startEvent));
        CUDA_CHECK(cudaEventDestroy(stopEvent));

        for (int idx = 0; idx < 2; ++idx) {
            CUDA_CHECK(cudaFree(d_posMass[idx]));
            CUDA_CHECK(cudaFree(d_vel[idx]));
        }

        timingSummary.emplace_back(filename, durationMs);
        appendTimingForFile(filename, durationMs);

        std::cout << "GPU simulation for " << filename << " finished in "
                  << durationMs << " ms." << std::endl;
        std::cout << "========================================" << std::endl;
    }

    writeTimingSummary(directoryPath, timingSummary);

    std::cout << "All GPU simulations complete." << std::endl;
    return 0;
}