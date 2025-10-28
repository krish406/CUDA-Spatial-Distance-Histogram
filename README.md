# CUDA Spatial Distance Histogram

This repository contains a CUDA implementation of a Spatial Distance Histogram (SDH) computation together with a CPU baseline. The implementation is in `proj2-krishs.cu` and computes a histogram of pairwise Euclidean distances between randomly distributed points.

## Overview

- The program generates N 3D points uniformly in a cube of size BOX_SIZE.
- It computes the pairwise distances for all unique pairs (i < j) and bins those distances into histogram buckets of width w (PDH_res).
- Two implementations are included:
  - `PDH_baseline`: brute-force, single-threaded CPU computation.
  - `PDH_baseline_parallel`: CUDA kernel with tiling and per-block shared-memory histograms for acceleration.

## Files

- `proj2-krishs.cu` — main program, CPU baseline, and CUDA kernel.

## Output

- The program prints the CPU histogram, GPU histogram, their element-wise difference, CPU runtime (gettimeofday), GPU kernel elapsed time (CUDA events), and the total count for verification (should be N*(N-1)/2).

## Optimization Strategies

- Data layout: Struct-of-Arrays (SoA) for coalesced GPU memory access using `atom_array` with separate x, y, and z arrays.
- Kernel: `PDH_baseline_parallel` uses shared memory to store a tile of points (R_x, R_y, R_z) and a per-block private histogram.
- Tiling strategy: each thread loads the left-side point and then iterates over right-side tiles loaded into shared memory to compute distances.
- Per-block histogram reduces contention on the global histogram; atomicAdd is used to safely increment counters in shared memory and then to merge local histograms into the global histogram.

