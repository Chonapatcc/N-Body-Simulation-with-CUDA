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
            } catch (const std::invalid_argument&) {
                std::cerr << "Warning: Bad data in " << filename << " on line: " << line << std::endl;
                values.clear();
                break;
            }
        }

        if (values.size() >= 7) {
            Particle p;
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

__global__ void nbodyStepKernel(const float* x_in,
                                const float* y_in,
                                const float* z_in,
                                const float* vx_in,
                                const float* vy_in,
                                const float* vz_in,
                                float* x_out,
                                float* y_out,
                                float* z_out,
                                float* vx_out,
                                float* vy_out,
                                float* vz_out,
                                const float* mass,
                                int N,
                                float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    float xi = x_in[i];
    float yi = y_in[i];
    float zi = z_in[i];
    float vxi = vx_in[i];
    float vyi = vy_in[i];
    float vzi = vz_in[i];
    float mi = mass[i];

    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;

    for (int j = 0; j < N; ++j) {
        if (i == j) continue;
        float dx = x_in[j] - xi;
        float dy = y_in[j] - yi;
        float dz = z_in[j] - zi;
        float dist_sq = dx * dx + dy * dy + dz * dz + EPSILON_CONST;
        float dist = sqrtf(dist_sq);
        float inv_dist_cubed = 1.0f / (dist_sq * dist);
        float force_scalar = G_CONST * mi * mass[j] * inv_dist_cubed;
        ax += force_scalar * dx / mi;
        ay += force_scalar * dy / mi;
        az += force_scalar * dz / mi;
    }

    vxi += ax * dt;
    vyi += ay * dt;
    vzi += az * dt;

    xi += vxi * dt;
    yi += vyi * dt;
    zi += vzi * dt;

    vx_out[i] = vxi;
    vy_out[i] = vyi;
    vz_out[i] = vzi;
    x_out[i] = xi;
    y_out[i] = yi;
    z_out[i] = zi;
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

        std::vector<float> h_x(N), h_y(N), h_z(N), h_vx(N), h_vy(N), h_vz(N), h_mass(N);
        for (int i = 0; i < N; ++i) {
            h_x[i] = particles[i].x;
            h_y[i] = particles[i].y;
            h_z[i] = particles[i].z;
            h_vx[i] = particles[i].vx;
            h_vy[i] = particles[i].vy;
            h_vz[i] = particles[i].vz;
            h_mass[i] = particles[i].mass;
        }

        const size_t bytes = static_cast<size_t>(N) * sizeof(float);

        float* d_x[2];
        float* d_y[2];
        float* d_z[2];
        float* d_vx[2];
        float* d_vy[2];
        float* d_vz[2];
        float* d_mass = nullptr;

        for (int idx = 0; idx < 2; ++idx) {
            CUDA_CHECK(cudaMalloc(&d_x[idx], bytes));
            CUDA_CHECK(cudaMalloc(&d_y[idx], bytes));
            CUDA_CHECK(cudaMalloc(&d_z[idx], bytes));
            CUDA_CHECK(cudaMalloc(&d_vx[idx], bytes));
            CUDA_CHECK(cudaMalloc(&d_vy[idx], bytes));
            CUDA_CHECK(cudaMalloc(&d_vz[idx], bytes));
        }
        CUDA_CHECK(cudaMalloc(&d_mass, bytes));

        CUDA_CHECK(cudaMemcpy(d_x[0], h_x.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y[0], h_y.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_z[0], h_z.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vx[0], h_vx.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vy[0], h_vy.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vz[0], h_vz.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mass, h_mass.data(), bytes, cudaMemcpyHostToDevice));

        std::cout << "Starting GPU N-Body simulation (naive global-memory kernel)..." << std::endl;

        cudaEvent_t startEvent{};
        cudaEvent_t stopEvent{};
        CUDA_CHECK(cudaEventCreate(&startEvent));
        CUDA_CHECK(cudaEventCreate(&stopEvent));
        CUDA_CHECK(cudaEventRecord(startEvent));

        int current = 0;
        int next = 1;

        const int threadsPerBlock = 256;
        const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

        for (int step = 0; step < STEPS; ++step) {
            nbodyStepKernel<<<blocks, threadsPerBlock>>>(
                d_x[current],
                d_y[current],
                d_z[current],
                d_vx[current],
                d_vy[current],
                d_vz[current],
                d_x[next],
                d_y[next],
                d_z[next],
                d_vx[next],
                d_vy[next],
                d_vz[next],
                d_mass,
                N,
                dt);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            if ((step + 1) % std::max(1, STEPS / 10) == 0) {
                float percent = (step + 1) * 100.0f / static_cast<float>(STEPS);
                std::cout << "\rProgress: " << std::fixed << std::setprecision(1)
                          << percent << "%" << std::flush;
            }

            std::swap(current, next);
        }
        std::cout << std::endl;

        CUDA_CHECK(cudaEventRecord(stopEvent));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));

        float elapsedMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
        long long durationMs = static_cast<long long>(std::llround(elapsedMs));

        CUDA_CHECK(cudaEventDestroy(startEvent));
        CUDA_CHECK(cudaEventDestroy(stopEvent));

        for (int idx = 0; idx < 2; ++idx) {
            CUDA_CHECK(cudaFree(d_x[idx]));
            CUDA_CHECK(cudaFree(d_y[idx]));
            CUDA_CHECK(cudaFree(d_z[idx]));
            CUDA_CHECK(cudaFree(d_vx[idx]));
            CUDA_CHECK(cudaFree(d_vy[idx]));
            CUDA_CHECK(cudaFree(d_vz[idx]));
        }
        CUDA_CHECK(cudaFree(d_mass));

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