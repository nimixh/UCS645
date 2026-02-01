# Q3: Calculation of Pi

Author: Nimish Badgujar (102497027)

## Problem Statement

Approximate pi using numerical integration:

pi = ∫(0 to 1) 4/(1+x²) dx

Implement using parallel reduction with OpenMP

## Implementation

Based on the pseudocode provided in the assignment:

```c
static long num_steps = 100000;
double step = 1.0 / (double)num_steps;
double sum = 0.0;

#pragma omp parallel for num_threads(threads) reduction(+:sum)
for (long i = 0; i < num_steps; i++) {
    double x = (i + 0.5) * step;
    sum = sum + 4.0 / (1.0 + x * x);
}

double pi = step * sum;
```

## Compilation

```bash
gcc -fopenmp Q3.c -o Q3 -O2
./Q3 <num_threads>
```

## Results

System: AMD Ryzen 5 4600H (6 cores / 12 threads)  
Steps: 100,000 (as per assignment)

| Threads | Time (s) | Computed Pi | Speedup | Notes |
|:-------:|:--------:|:-----------:|:-------:|:-----:|
| 1 | 0.000270 | 3.141592653598162 | 1.00x | Baseline |
| 2 | 0.000233 | 3.141592653598146 | 1.16x | |
| 4 | 0.000239 | 3.141592653598127 | 1.13x | |
| 6 | 0.000247 | 3.141592653598132 | 1.09x | |
| 8 | 0.000568 | 3.141592653598125 | 0.48x | |
| 12 | 0.000432 | 3.141592653598126 | 0.63x | Hardware limit |
| 16 | 0.000598 | 3.141592653598125 | 0.45x | Oversubscribed |
| 24 | 0.000808 | 3.141592653598127 | 0.33x | Oversubscribed |

Accuracy: Pi correct to 11+ decimal places

![Q3 Speedup Graph](q3_speedup.png)

## Analysis

### Which thread count gives maximum speedup?

2 threads provides the best speedup (1.16x) for this workload.

### Why Limited Scaling?

1. Small Problem Size: 100,000 steps is too small - thread overhead dominates
2. Reduction Overhead: Combining partial sums requires synchronization
3. Embarrassingly Parallel but Tiny: Each iteration is independent, but so fast that parallelization overhead exceeds benefit

### What Happens When Threads Increase Beyond Optimal?

Performance degrades:

At 8+ threads:
- Time increases significantly
- Thread creation + synchronization completely dominates
- Overhead exceeds any parallel benefit

Beyond 12 threads (oversubscription):
- 16/24 threads: Continue to perform poorly
- Context switching adds more overhead
- Still slower than single-threaded

### Mathematical Background

The integral uses the identity:

∫(0 to 1) 4/(1+x²) dx = 4·arctan(x)|(0 to 1) = 4·(pi/4 - 0) = pi

Using midpoint rule: x_i = (i + 0.5) × Δx where Δx = 1/num_steps

### Key Insight

For tiny workloads, even "embarrassingly parallel" algorithms suffer from parallelization overhead. The problem must be large enough to amortize thread management costs.

### Improving Performance

To see better speedup, increase num_steps:
- 100,000 -> ~0.0002s (overhead dominated)
- 10,000,000 -> ~0.02s (better scaling possible)
- 1,000,000,000 -> ~2s (near-linear speedup achievable)
