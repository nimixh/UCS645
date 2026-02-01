/*
 * Q2 Part B: Matrix Multiplication with 2D Threading (collapse)
 * Course: UCS645 - Parallel and Distributed Computing
 * Author: Nimish Badgujar (102497027)
 *
 * Parallelizes both i and j loops using collapse(2)
 * Matrix size: 1000 x 1000
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000  // Matrix dimension as per assignment

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <num_threads>\n", argv[0]);
        return 1;
    }

    int threads = atoi(argv[1]);

    // Using static to allocate on BSS segment (avoids stack overflow)
    static double A[N][N], B[N][N], C[N][N];

    // Initialize matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = 1.0;
            B[i][j] = 1.0;
            C[i][j] = 0.0;
        }
    }

    double start = omp_get_wtime();

    // 2D parallelization: collapse both i and j loops
    #pragma omp parallel for collapse(2) num_threads(threads)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    double end = omp_get_wtime();

    printf("[2D] Threads: %d | Time: %f seconds\n", threads, end - start);

    return 0;
}
