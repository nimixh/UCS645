#include <cmath>
#include <omp.h>


void correlate_seq(int ny, int nx, const float* data, float* result) {

    double* norm = new double[ny * nx];

    // Normalize rows
    for (int y = 0; y < ny; y++) {
        double mean = 0.0;
        for (int x = 0; x < nx; x++)
            mean += data[x + y * nx];
        mean /= nx;

        double sq = 0.0;
        for (int x = 0; x < nx; x++) {
            norm[x + y * nx] = data[x + y * nx] - mean;
            sq += norm[x + y * nx] * norm[x + y * nx];
        }

        double inv = sq > 0.0 ? 1.0 / std::sqrt(sq) : 0.0;
        for (int x = 0; x < nx; x++)
            norm[x + y * nx] *= inv;
    }

    // Correlation
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int x = 0; x < nx; x++)
                sum += norm[x + i * nx] * norm[x + j * nx];

            result[i + j * ny] = static_cast<float>(sum);
        }
    }

    delete[] norm;
}


void correlate_omp(int ny, int nx, const float* data, float* result) {

    double* norm = new double[ny * nx];

    #pragma omp parallel for
    for (int y = 0; y < ny; y++) {
        double mean = 0.0;
        for (int x = 0; x < nx; x++)
            mean += data[x + y * nx];
        mean /= nx;

        double sq = 0.0;
        for (int x = 0; x < nx; x++) {
            norm[x + y * nx] = data[x + y * nx] - mean;
            sq += norm[x + y * nx] * norm[x + y * nx];
        }

        double inv = sq > 0.0 ? 1.0 / std::sqrt(sq) : 0.0;
        for (int x = 0; x < nx; x++)
            norm[x + y * nx] *= inv;
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int x = 0; x < nx; x++)
                sum += norm[x + i * nx] * norm[x + j * nx];

            result[i + j * ny] = static_cast<float>(sum);
        }
    }

    delete[] norm;
}


void correlate_fast(int ny, int nx, const float* data, float* result) {

    double* norm = new double[ny * nx];

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < ny; y++) {
        double mean = 0.0;
        for (int x = 0; x < nx; x++)
            mean += data[x + y * nx];
        mean /= nx;

        double sq = 0.0;
        for (int x = 0; x < nx; x++) {
            norm[x + y * nx] = data[x + y * nx] - mean;
            sq += norm[x + y * nx] * norm[x + y * nx];
        }

        double inv = sq > 0.0 ? 1.0 / std::sqrt(sq) : 0.0;
        for (int x = 0; x < nx; x++)
            norm[x + y * nx] *= inv;
    }

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int x = 0; x < nx; x++)
                sum += norm[x + i * nx] * norm[x + j * nx];

            result[i + j * ny] = static_cast<float>(sum);
        }
    }

    delete[] norm;
}
