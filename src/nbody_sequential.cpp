#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <filesystem> // For directory scanning (requires C++17)
#include <fstream>    // For reading files
#include <sstream>    // For parsing file lines
#include <iomanip>

// --- Constants and Structs (Unchanged) ---
const float G = 6.67430e-11f;
const float EPSILON = 1e-3f;

struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float mass;
};

struct Acceleration {
    float ax, ay, az;
};

/**
 * UPDATED FUNCTION: loadParticlesFromCSV
 * Loads particle data from a specified CSV file.
 */
int loadParticlesFromCSV(std::vector<Particle>& particles, const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return -1;
    }

    std::string line;
    bool is_header = true; // Flag to skip the first line (header)
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
            } catch (const std::invalid_argument& e) {
                std::cerr << "Warning: Bad data in " << filename << " on line: " << line << std::endl;
                values.clear(); 
                break;
            }
        }

        // --- THIS IS THE ONLY LINE I CHANGED ---
        // We now accept 7 or more columns, to safely ignore extra
        // columns like particle IDs.
        if (values.size() >= 7) { 
        // -------------------------------------
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
             // This warning for short lines (like the 4-column one) is still correct.
             std::cerr << "Warning: Skipping malformed line in " << filename << " (expected >= 7 columns): " << line << std::endl;
        }
    }

    file.close();
    std::cout << "Successfully loaded " << n_loaded << " particles from " << filename << std::endl;
    return n_loaded; // Return the number of particles
}


// --- calculateForces (Unchanged) ---
void calculateForces(std::vector<Particle>& particles, std::vector<Acceleration>& acc, int N) {
    for (int i = 0; i < N; ++i) {
        float total_fx = 0.0f;
        float total_fy = 0.0f;
        float total_fz = 0.0f;

        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            float dx = particles[j].x - particles[i].x;
            float dy = particles[j].y - particles[i].y;
            float dz = particles[j].z - particles[i].z;
            float dist_sq = dx * dx + dy * dy + dz * dz + EPSILON;
            float dist = std::sqrt(dist_sq);
            float inv_dist_cubed = 1.0f / (dist_sq * dist);
            float force_scalar = G * particles[i].mass * particles[j].mass * inv_dist_cubed;
            total_fx += force_scalar * dx;
            total_fy += force_scalar * dy;
            total_fz += force_scalar * dz;
        }
        acc[i].ax = total_fx / particles[i].mass;
        acc[i].ay = total_fy / particles[i].mass;
        acc[i].az = total_fz / particles[i].mass;
    }
}

// --- updateParticles (Unchanged) ---
void updateParticles(std::vector<Particle>& particles, const std::vector<Acceleration>& acc, int N, float dt) {
    for (int i = 0; i < N; ++i) {
        particles[i].vx += acc[i].ax * dt;
        particles[i].vy += acc[i].ay * dt;
        particles[i].vz += acc[i].az * dt;
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
        particles[i].z += particles[i].vz * dt;
    }
}


// --- getCSVFiles (Unchanged) ---
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


// --- main() (Unchanged) ---
int main() {
    const float dt = 0.01f; // Time step
    const int STEPS = 10;
    const std::string directoryPath = "data/star_cluster_simulation"; 

    std::vector<std::string> csvFiles = getCSVFiles(directoryPath);

    if (csvFiles.empty()) {
        std::cerr << "No .csv files found in " << directoryPath << std::endl;
        return 1;
    }

    std::cout << "Found " << csvFiles.size() << " files to process." << std::endl;
    std::cout << "========================================" << std::endl;

    for (const std::string& filename : csvFiles) {
        
        std::vector<Particle> particles; 
        int N = loadParticlesFromCSV(particles, filename);
        
        if (N <= 0) {
            std::cerr << "Failed to load " << filename << " or file is empty. Skipping." << std::endl;
            std::cout << "----------------------------------------" << std::endl;
            continue; 
        }

        std::vector<Acceleration> accelerations(N);

        std::cout << "Starting sequential N-Body simulation..." << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < STEPS; ++step) {
            calculateForces(particles, accelerations, N);
            updateParticles(particles, accelerations, N, dt);
            if ((step + 1) % std::max(1, STEPS / 10) == 0) {
                float percent = (step + 1) * 100.0f / STEPS;
                std::cout << "\rProgress: " << std::fixed << std::setprecision(1) << percent << "%";
            }
        }
        std::cout << std::endl;

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        
        std::cout << "Sequential simulation for " << filename << " finished in " 
                  << duration.count() << " ms." << std::endl;
        std::cout << "========================================" << std::endl;

    } 

    std::cout << "All simulations complete." << std::endl;
    return 0;
}