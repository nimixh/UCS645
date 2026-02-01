/*
 * Author: Nimish Badgujar
 * Roll No: 102497027
 * Group: 3Q21
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static long num_steps = 100000;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <num_threads>\n", argv[0]);
        return 1;
    }

    int threads = atoi(argv[1]);
    double step = 1.0 / (double)num_steps;
    double sum = 0.0;

    double start = omp_get_wtime();

    #pragma omp parallel for num_threads(threads) reduction(+:sum)
    for (long i = 0; i < num_steps; i++) {
        double x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }

    double pi = step * sum;
    double end = omp_get_wtime();

    printf("Threads: %d | Pi: %.15f | Time: %f seconds\n", threads, pi, end - start);

    return 0;
}
