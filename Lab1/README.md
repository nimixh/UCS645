# 🧪 Lab 1: OpenMP Fundamentals

> **Exploring thread-level parallelism with OpenMP on AMD Ryzen 5 4600H**

---

## 💻 System Configuration

### Processor Details

| Specification       | Value                                  |
| ------------------- | -------------------------------------- |
| **CPU**             | AMD Ryzen 5 4600H with Radeon Graphics |
| **Architecture**    | x86_64 (Zen 2)                         |
| **Physical Cores**  | 6                                      |
| **Logical Threads** | 12                                     |
| **SMT**             | 2 threads per core                     |
| **Base Clock**      | ~3.0 GHz                               |
| **Boost Clock**     | Up to 4.0 GHz                          |

### Cache Hierarchy

| Level              | Size    | Instances |
| ------------------ | ------- | --------- |
| **L1 Data**        | 192 KiB | 6         |
| **L1 Instruction** | 192 KiB | 6         |
| **L2**             | 3 MiB   | 6         |
| **L3**             | 8 MiB   | 2         |

### Development Environment

| Component    | Version       |
| ------------ | ------------- |
| **OS**       | Arch Linux    |
| **Kernel**   | Native Linux  |
| **Compiler** | GCC 15.2.1    |
| **OpenMP**   | libgomp (GCC) |

```bash
$ gcc --version
gcc (GCC) 15.2.1 20260103
Copyright (C) 2025 Free Software Foundation, Inc.
```

---

## 📁 Lab Contents

| Question | Topic                 | Description                          |
| -------- | --------------------- | ------------------------------------ |
| Q1       | DAXPY                 | Vector operation parallelization     |
| Q2       | Matrix Multiplication | 1D and 2D parallel strategies        |
| Q3       | Pi Approximation      | Numerical integration with reduction |
