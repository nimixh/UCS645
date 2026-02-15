#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <random>

void correlate_seq(int ny, int nx, const float* data, float* result);
void correlate_omp(int ny, int nx, const float* data, float* result);
void correlate_fast(int ny, int nx, const float* data, float* result);

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: ./correlate <ny> <nx>\n";
        return 1;
    }

    int ny = std::atoi(argv[1]);
    int nx = std::atoi(argv[2]);

    float* data = new float[ny * nx];
    float* result = new float[ny * ny];

    // Fill matrix with random values
    std::mt19937 rng(102497027);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < ny * nx; i++)
        data[i] = dist(rng);

    std::cout << "Running Sequential...\n";
    auto start = std::chrono::high_resolution_clock::now();
    correlate_seq(ny, nx, data, result);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Sequential Time: "
              << std::chrono::duration<double>(end - start).count()
              << " sec\n\n";

    std::cout << "Running OpenMP...\n";
    start = std::chrono::high_resolution_clock::now();
    correlate_omp(ny, nx, data, result);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "OpenMP Time: "
              << std::chrono::duration<double>(end - start).count()
              << " sec\n\n";

    std::cout << "Running Fast Version...\n";
    start = std::chrono::high_resolution_clock::now();
    correlate_fast(ny, nx, data, result);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Fast Time: "
              << std::chrono::duration<double>(end - start).count()
              << " sec\n";

    delete[] data;
    delete[] result;

    return 0;
}
