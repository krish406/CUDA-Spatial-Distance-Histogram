# CUDA Spatial Distance Histogram

This repository contains a CUDA implementation of a Spatial Distance Histogram (SDH) computation together with a CPU baseline. The implementation is in `proj2-krishs.cu` and computes a histogram of pairwise Euclidean distances between points randomly distributed inside a 3D box.

## Overview

- The program generates N 3D points uniformly in a cube of size BOX_SIZE (defined in the source as 23000).
- It computes the pairwise distances for all unique pairs (i < j) and bins those distances into histogram buckets of width w (PDH_res).
- Two implementations are included:
  - `PDH_baseline`: brute-force, single-threaded CPU computation.
  - `PDH_baseline_parallel`: CUDA kernel with tiling and per-block shared-memory histograms for acceleration.

## Files

- `proj2-krishs.cu` — main program, CPU baseline, and CUDA kernel.

## Build

You need the NVIDIA CUDA toolkit and `nvcc`.

Example compile command:

```
vcc -O3 proj2-krishs.cu -o sdh
```

## Usage

Run the executable with three arguments:

```
./sdh <num_points> <bucket_width> <block_dim>
```

- `num_points` (integer): number of points to generate (PDH_acnt).
- `bucket_width` (float): bucket size (PDH_res).
- `block_dim` (integer): CUDA threads per block (must be >0, <=1024, and a multiple of 32).

Example:

```
./sdh 10000 1.0 256
```

## Output

- The program prints the CPU histogram, GPU histogram, their element-wise difference, CPU runtime (gettimeofday), GPU kernel elapsed time (CUDA events), and the total count for verification (should be N*(N-1)/2).

## Implementation highlights

- Data layout: Struct-of-Arrays (SoA) for coalesced GPU memory access using `atom_array` with separate `x_pos`, `y_pos`, `z_pos` arrays.
- Kernel: `PDH_baseline_parallel` uses shared memory to store a tile of points (R_x, R_y, R_z) and a per-block private histogram `local_hist`.
- Tiling strategy: each thread loads the left-side point and then iterates over right-side tiles loaded into shared memory to compute distances.
- Per-block histogram reduces contention on the global histogram; atomicAdd is used to safely increment counters in shared memory and then to merge local histograms into the global histogram.
- Distances and counters use double precision and 64-bit counters respectively.

## Important notes and limitations

- Shared memory footprint: the kernel allocates shared memory sized as `3 * BLOCK_DIM * sizeof(double) + sizeof(bucket) * num_buckets`. If `num_buckets` is large, this may exceed device shared memory limits and cause kernel launch failures or incorrect behavior.
- The current approach stores the entire histogram per-block in shared memory; for large bucket counts this is not scalable. Consider processing the histogram in ranges (multi-pass) or using smaller per-block partial histograms.
- 64-bit atomicAdd requirements: the code casts to `(unsigned long long *)` for 64-bit atomicAdd; ensure the target GPU architecture supports 64-bit global atomics.
- The code uses doubles for coordinates and distances; switching to floats can reduce memory and compute overhead if precision allows.

## Verification

- The program prints both CPU and GPU histograms and their difference. The total count across buckets should equal N*(N-1)/2.

## Suggestions for improvements

- Reduce shared-memory usage for per-block histogram by using histogram chunking (multi-pass), per-warp histograms, or warp-level reductions.
- Replace some atomics with per-thread or per-warp private histograms and do a block-level reduction to reduce atomic contention.
- Use ceiling division when computing block count: `(PDH_acnt + BLOCK_DIM - 1) / BLOCK_DIM`.
- Add explicit zero-initialization of `gpu_histogram` before copying to the device to avoid relying on device memory state.

## Reference

See the implementation in `proj2-krishs.cu` for exact details and implementation choices:
https://github.com/krish406/CUDA-Spatial-Distance-Histogram/blob/main/proj2-krishs.cu
```
