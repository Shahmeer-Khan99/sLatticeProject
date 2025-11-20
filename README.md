# Parallel Algorithms in CUDA/C++: Minimum Search and TCN Simplification

This project implements two high-performance parallel algorithms in C++ and CUDA as part of the **Master in HPC / Lattice Theory for Parallel Programming** course at the **University of Luxembourg**.

---

## Project Overview

This project covers:

1. **Exercise 1 – Parallel Minimum**: Compute the minimum value in a large array using both CPU and GPU parallel algorithms.  
2. **Exercise 2 – Parallel Simplification of Ternary Constraint Networks (TCNs)**: Simplify logical formulas by detecting equivalent variables and deprecating redundant constraints using GPU parallelization.

---

## Exercise 1: Parallel Minimum

### Implementation

- **Sequential Algorithm**: Standard linear scan on CPU.  
- **Global Atomic Kernel**: Each GPU thread updates a global minimum using `atomicMin`.  
- **Reduction Kernel**: Optimized block-wise reduction using shared memory, minimizing divergence and maximizing parallel efficiency.

### Usage

```bash
# Compile
nvcc -O3 min_bench.cu -o min_bench
```

# Run
```bash
./min_bench [N] [mode]
# N     : number of elements (default 1<<20)
# mode  : 0=sequential, 1=global_atomic, 2=reduction, 3=all (default 3)
```

# Results

## Exercise 1: Array Minimum Finding

| Implementation | N (elements) | Blocks | Threads | Time (ms) | Minimum Value |
|---------------|-------------|--------|---------|-----------|---------------|
| Sequential | 1,048,576 | - | - | 4.61 | -1,000,000 |
| Global Atomic | 1,048,576 | 1024 | 1024 | 5.86 | -1,000,000 |
| Reduction | 1,048,576 | auto | 1024 | 0.073 | -1,000,000 |

**Observation:** The reduction kernel is over 60× faster than sequential CPU computation and significantly faster than the global atomic approach due to minimized atomic contention and efficient shared memory use.

## Exercise 2: Parallel Simplification of Logical Formula

### Implementation

- **Variables and Constraints:** Represented as `Variable` and `Constraint` structs
- **Parallel Union-Find:** GPU implementation to merge equivalent variables efficiently
- **Fixpoint Computation:** Iterative GPU kernel `gpu_merge_constraints` applies simplifications until no changes occur
- **Constraint Deprecation:** Redundant constraints are marked as `R` and skipped in further computation

### Usage

```bash
# Compile
nvcc -O3 simplify.cu -o simplify

# Run
./simplify -o output.tcn input.tcn

# Results

## Input File Statistics

| Input File | N (variables) | M (constraints) |
|------------|--------------|----------------|
| yumi-dynamic_p_4_GG_GG_yumi_grid_setup_5_5_zones.tcn | 9,476,831 | 12,344,286 |
| yumi-dynamic_p_5_GG_GGG_yumi_grid_setup_5_5_zones.tcn | 11,494,550 | 14,876,065 |
| aircraft-disassembly_B737NG-600-09-Anon.tcn | 5,671,645 | 5,884,161 |
| portal_small_3.tcn | 93,489 | 121,941 |

**Observation:** The GPU-based parallel simplification handles millions of variables and constraints efficiently. Equivalence detection and redundant constraint removal significantly reduce the network complexity.

# Conclusion

- **Exercise 1:** GPU reduction outperforms both sequential and global atomic methods for large arrays, achieving massive speedups

- **Exercise 2:** Parallel fixpoint computation and GPU union-find enable fast simplification of massive ternary constraint networks. Both correctness and performance scale well with input size

# Dependencies

- CUDA Toolkit 11+
- C++17 compatible compiler
- Linux/Unix environment
