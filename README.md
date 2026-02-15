# Assignment 3: Parallel Correlation Matrix Computation

## Student Details

- Name: Nimish Badgujar
- Roll No: 102497027
- Group: 3Q21

## Problem Statement

Given a matrix of size **ny × nx** containing **m input vectors (rows)**,  
compute the **correlation coefficient** between every pair of input vectors.

For all `0 ≤ j ≤ i < ny`, compute:

    correlation(row_i, row_j)

Store the result in the lower triangular matrix location:

    result[i + j * ny]

All arithmetic operations are performed using **double precision**.

---

## Implementations

Three versions are implemented:

| Version      | Description                                           |
|--------------|-------------------------------------------------------|
| Sequential   | Baseline single-threaded implementation               |
| OpenMP       | Multi-threaded parallel implementation                |
| Fast         | OpenMP + SIMD + -O3 compiler optimizations            |

---

## Small Code Updates

- Input generation is now deterministic using a fixed random seed for reproducible benchmarking.
- Row normalization now includes a zero-variance safeguard to avoid division-by-zero.

---

## Compilation

```bash
make
```

---

## Execution

```bash
./correlate <ny> <nx>
```

Example:

```bash
./correlate 2000 2000
```

---

# Performance Evaluation

**Matrix Size Tested:** 2000 × 2000

---

## Execution Time Comparison

### 1 Thread

| Version     | Time (sec) |
|------------|------------|
| Sequential | 3.82343    |
| OpenMP     | 3.82358    |
| Fast       | 3.89792    |

### 2 Threads

| Version     | Time (sec) |
|------------|------------|
| Sequential | 3.83916    |
| OpenMP     | 2.42978    |
| Fast       | 2.01104    |

### 4 Threads

| Version     | Time (sec) |
|------------|------------|
| Sequential | 3.79416    |
| OpenMP     | 2.04016    |
| Fast       | 2.20233    |

### 8 Threads

| Version     | Time (sec) |
|------------|------------|
| Sequential | 3.77409    |
| OpenMP     | 2.24585    |
| Fast       | 2.06626    |

---

# Speedup Analysis (Fast Version)

Speedup formula:

    Speedup = T1 / Tn

| Threads | Fast Time (sec) | Speedup |
|---------|-----------------|----------|
| 1       | 3.89792         | 1.00×    |
| 2       | 2.01104         | 1.94×    |
| 4       | 2.20233         | 1.77×    |
| 8       | 2.06626         | 1.89×    |

Parallel Efficiency (8 threads):

    Efficiency = 1.89 / 8 ≈ 23.6%

---

#  perf Statistics (8 Threads)

| Metric                     | Value     |
|----------------------------|-----------|
| CPUs utilized              | ~2.43     |
| Instructions per cycle     | ~1.27     |
| Branch miss rate           | ~0.08%    |
| Total elapsed time         | ~6.05 sec |

---

#  Optimization Techniques Used

- Precomputation of row means and normalization  
- Lower triangular computation only  
- OpenMP parallelization  
- Loop collapse for nested loops  
- SIMD reduction pragma  
- Compiler optimization flag -O3  
- Improved memory access locality  

---

#  Observations

- Time complexity grows approximately O(ny² × nx).
- Parallel implementation significantly reduces execution time.
- Speedup in this run is modest and non-monotonic across thread counts.
- Memory bandwidth begins to limit scaling at higher thread counts.
- SIMD improves inner loop arithmetic throughput.

---

##  Clean Build

```bash
make clean
```

