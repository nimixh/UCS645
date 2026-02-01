/*
 * Q1: DAXPY Operation (Double precision A*X Plus Y)
 * Course: UCS645 - Parallel and Distributed Computing
 * Author: Nimish Badgujar (102497027)
 *
 * Operation: X[i] = a * X[i] + Y[i]
 * Vector size: 2^16 = 65536 elements
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N (1 << 16)  // 2^16 as per assignment

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <num_threads>\n", argv[0]);
        return 1;
    }

    int threads = atoi(argv[1]);
    
    double *X = malloc(N * sizeof(double));
    double *Y = malloc(N * sizeof(double));
    double a = 2.5;  // Scalar value for DAXPY

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        X[i] = 1.0;
        Y[i] = 2.0;
    }

    double start = omp_get_wtime();

    // DAXPY: X[i] = a * X[i] + Y[i]
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < N; i++) {
        X[i] = a * X[i] + Y[i];
    }

    double end = omp_get_wtime();

    printf("Threads: %d | Time: %f seconds\n", threads, end - start);

    free(X);
    free(Y);

    return 0;
}


