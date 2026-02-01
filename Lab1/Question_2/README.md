# Q2: Matrix Multiplication

Author: Nimish Badgujar (102497027)

## Problem Statement

Build parallel matrix multiplication for 1000x1000 matrices  
Implement two versions:
1. 1D Threading: Parallelize single (outer) loop
2. 2D Threading: Use collapse(2) to parallelize nested loops

## Implementation

### Part A - 1D Threading (Q2.c)

```c
#pragma omp parallel for num_threads(threads)
for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
        for (int k = 0; k < N; k++)
            C[i][j] += A[i][k] * B[k][j];
```

### Part B - 2D Threading (Q2b.c)

```c
#pragma omp parallel for collapse(2) num_threads(threads)
for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
        for (int k = 0; k < N; k++)
            C[i][j] += A[i][k] * B[k][j];
```

## Compilation

```bash
cd Part_1 && gcc -fopenmp Q2.c -o Q2 -O2
cd Part_2 && gcc -fopenmp Q2b.c -o Q2b -O2
```

## Results - 1D Threading

System: AMD Ryzen 5 4600H (6 cores / 12 threads)  
Matrix Size: 1000 x 1000

| Threads | Time (s) | Speedup | Efficiency | Notes |
|:-------:|:--------:|:-------:|:----------:|:-----:|
| 1 | 1.049611 | 1.00x | 100.0% | Baseline |
| 2 | 0.671124 | 1.56x | 78.2% | |
| 4 | 0.345118 | 3.04x | 76.0% | |
| 6 | 0.235986 | 4.45x | 74.1% | |
| 8 | 0.173409 | 6.05x | 75.7% | |
| 12 | 0.159662 | 6.57x | 54.8% | Hardware limit |
| 16 | 0.180315 | 5.82x | 36.4% | Oversubscribed |
| 24 | 0.192184 | 5.46x | 22.8% | Oversubscribed |

![1D Speedup](q2_1d_speedup.png)

## Results - 2D Threading (collapse)

| Threads | Time (s) | Speedup | Efficiency | Notes |
|:-------:|:--------:|:-------:|:----------:|:-----:|
| 1 | 1.351594 | 1.00x | 100.0% | Baseline |
| 2 | 0.810649 | 1.67x | 83.4% | |
| 4 | 0.405591 | 3.33x | 83.3% | |
| 6 | 0.288510 | 4.68x | 78.1% | |
| 8 | 0.209516 | 6.45x | 80.6% | |
| 12 | 0.250924 | 5.39x | 44.9% | Hardware limit |
| 16 | 0.214779 | 6.29x | 39.3% | Oversubscribed |
| 24 | 0.228982 | 5.90x | 24.6% | Oversubscribed |

![2D Speedup](q2_2d_speedup.png)

## Analysis

### Work Partitioning Strategy

1D (Row Distribution):
- Each thread processes N/threads complete rows
- Good cache locality for row access
- Simple scheduling with low overhead

2D (Collapsed Distribution):
- Iteration space = N x N = 1,000,000 iterations
- More balanced work distribution
- Better scalability at higher thread counts

### Comparison

| Metric | 1D | 2D |
|--------|----|----|
| Best 1-thread time | 1.05s | 1.35s |
| Best thread count | 12 | 8 |
| Best time | 0.16s @ 12T | 0.21s @ 8T |
| Max speedup | 6.57x @ 12T | 6.45x @ 8T |

### Key Findings

1. 1D has lower overhead - Faster single-threaded performance
2. Both show good scaling up to 8 threads
3. Performance plateaus at 12 threads (hardware limit)
4. Oversubscription effects visible at 16/24 threads

### What Happens When Threads Increase Beyond Optimal Point?

Optimal: 8-12 threads depending on implementation

Beyond 12 threads (Oversubscription):
1. Context Switching: CPU must time-slice threads onto physical cores
2. Cache Thrashing: Threads constantly evict each other's cache data
3. Scheduler Overhead: OS spends cycles managing threads instead of doing work
4. Memory Bandwidth Saturation: Already maxed out, no improvement possible

Observable Effects:
- 16 threads: Performance degrades by 10-13%
- 24 threads: Performance degrades by 15-17%
