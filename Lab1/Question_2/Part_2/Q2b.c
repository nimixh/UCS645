/*
 * Author: Nimish Badgujar
 * Roll No: 102497027
 * Group: 3Q21
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <num_threads>\n", argv[0]);
        return 1;
    }

    int threads = atoi(argv[1]);

    static double A[N][N], B[N][N], C[N][N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = 1.0;
            B[i][j] = 1.0;
            C[i][j] = 0.0;
        }
    }

    double start = omp_get_wtime();

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
