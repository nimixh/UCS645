# Q1: DAXPY Operation

## Problem Statement

> **DAXPY**: Double precision A*X Plus Y  
> Compute `X[i] = a*X[i] + Y[i]` for vectors of size 2^16 (65,536)  
> Compare speedup gained by varying thread count starting from 2 threads

---

## Implementation

```c
#pragma omp parallel for num_threads(threads)
for (int i = 0; i < N; i++) {
    X[i] = a * X[i] + Y[i];
}
```

### Compilation

```bash
gcc -fopenmp Q1.c -o Q1 -O2
./Q1 <num_threads>
```

---

## Results

**System**: AMD Ryzen 5 4600H (6 cores / 12 threads)  
**Vector Size**: 65,536 elements (2^16)

| Threads | Time (s) | Speedup | Efficiency |
| :-----: | :------: | :-----: | :--------: |
|    1    | 0.000028 |  1.00×  |   100.0%   |
|    2    | 0.000176 |  0.16×  |    8.0%    |
|    4    | 0.000158 |  0.18×  |    4.4%    |
|    6    | 0.000207 |  0.14×  |    2.3%    |
|    8    | 0.000240 |  0.12×  |    1.5%    |
|   12    | 0.000967 |  0.03×  |    0.2%    |

### Visualization

![Q1 Speedup Graph](q1_speedup.png)

---

## Analysis

### Which thread count gives maximum speedup?

**1 thread gives the best performance.** For this small, memory-bound operation, parallelization introduces overhead that exceeds any potential benefit.

### What happens when threads increase beyond optimal?

Performance **degrades significantly** because:

1. **Thread Creation Overhead**: Creating/destroying threads costs more than the computation itself
2. **Memory Bandwidth Saturation**: DAXPY is memory-bound (2 FLOPs per 24 bytes accessed)
3. **Cache Contention**: Multiple threads compete for memory bandwidth and cache lines
4. **Small Workload**: Only ~5,461 elements per thread at 12 threads - insufficient to amortize overhead

### Key Insight

DAXPY demonstrates that **not all loops benefit from parallelization**. For small, memory-bound kernels, sequential execution is optimal.
