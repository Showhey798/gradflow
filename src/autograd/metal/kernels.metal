// src/autograd/metal/kernels.metal
// Metal Compute Shaders for GradFlow
// Copyright (c) 2025 GradFlow Project
//
// This file contains Metal Shading Language kernels for
// elementwise operations, reduction operations, and matrix operations.

#include <metal_stdlib>
using namespace metal;

// ===================================================================
// Elementwise Operations
// ===================================================================

/**
 * @brief 要素ごとの加算カーネル: c = a + b
 *
 * Thread Group Size: 256
 * Grid Size: (size + 255) / 256
 *
 * @param a 入力配列 A [[buffer(0)]]
 * @param b 入力配列 B [[buffer(1)]]
 * @param c 出力配列 C [[buffer(2)]]
 * @param size 要素数 [[buffer(3)]]
 * @param gid スレッドの global position
 */
kernel void add_kernel(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       constant uint& size [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
  if (gid < size) {
    c[gid] = a[gid] + b[gid];
  }
}

/**
 * @brief 要素ごとの乗算カーネル: c = a * b
 *
 * Thread Group Size: 256
 * Grid Size: (size + 255) / 256
 *
 * @param a 入力配列 A [[buffer(0)]]
 * @param b 入力配列 B [[buffer(1)]]
 * @param c 出力配列 C [[buffer(2)]]
 * @param size 要素数 [[buffer(3)]]
 * @param gid スレッドの global position
 */
kernel void mul_kernel(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       constant uint& size [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
  if (gid < size) {
    c[gid] = a[gid] * b[gid];
  }
}

/**
 * @brief 要素ごとの減算カーネル: c = a - b
 *
 * Thread Group Size: 256
 * Grid Size: (size + 255) / 256
 *
 * @param a 入力配列 A [[buffer(0)]]
 * @param b 入力配列 B [[buffer(1)]]
 * @param c 出力配列 C [[buffer(2)]]
 * @param size 要素数 [[buffer(3)]]
 * @param gid スレッドの global position
 */
kernel void sub_kernel(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       constant uint& size [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
  if (gid < size) {
    c[gid] = a[gid] - b[gid];
  }
}

/**
 * @brief 要素ごとの除算カーネル: c = a / b
 *
 * Thread Group Size: 256
 * Grid Size: (size + 255) / 256
 *
 * @param a 入力配列 A [[buffer(0)]]
 * @param b 入力配列 B [[buffer(1)]]
 * @param c 出力配列 C [[buffer(2)]]
 * @param size 要素数 [[buffer(3)]]
 * @param gid スレッドの global position
 */
kernel void div_kernel(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       constant uint& size [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
  if (gid < size) {
    c[gid] = a[gid] / b[gid];
  }
}

// ===================================================================
// Reduction Operations
// ===================================================================

/**
 * @brief 総和カーネル (Stage 1: Thread group ごとに local sum を計算)
 *
 * Thread Group Size: 256
 * Threadgroup Memory: 256 * sizeof(float)
 *
 * このカーネルは、大きな配列を thread group ごとに分割し、
 * 各 thread group 内で parallel reduction を実行して partial sum を計算します。
 *
 * @param input 入力配列 [[buffer(0)]]
 * @param partial_sums 部分和の出力配列 [[buffer(1)]]
 * @param size 入力配列の要素数 [[buffer(2)]]
 * @param local_sum Threadgroup memory [[threadgroup(0)]]
 * @param gid Thread position in grid
 * @param tid Thread position in threadgroup
 * @param tpg Threads per threadgroup
 */
kernel void sum_kernel_stage1(
    device const float* input [[buffer(0)]],
    device float* partial_sums [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    threadgroup float* local_sum [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]) {
  // Load data into threadgroup memory
  local_sum[tid] = (gid < size) ? input[gid] : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Parallel reduction in threadgroup
  for (uint stride = tpg / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      local_sum[tid] += local_sum[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write partial sum (thread 0 in each threadgroup writes the result)
  if (tid == 0) {
    uint group_id = gid / tpg;
    partial_sums[group_id] = local_sum[0];
  }
}

/**
 * @brief 総和カーネル (Stage 2: partial sums を集約)
 *
 * Thread Group Size: 256
 * Threadgroup Memory: 256 * sizeof(float)
 *
 * Stage 1 で計算された partial sums を最終的に集約して、
 * 全体の総和を計算します。
 *
 * @param partial_sums Stage 1 からの部分和配列 [[buffer(0)]]
 * @param output 最終的な総和 [[buffer(1)]]
 * @param num_partials 部分和の個数 [[buffer(2)]]
 * @param local_sum Threadgroup memory [[threadgroup(0)]]
 * @param tid Thread position in threadgroup
 * @param tpg Threads per threadgroup
 */
kernel void sum_kernel_stage2(
    device const float* partial_sums [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& num_partials [[buffer(2)]],
    threadgroup float* local_sum [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]) {
  // Load partial sums
  local_sum[tid] = (tid < num_partials) ? partial_sums[tid] : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Parallel reduction
  for (uint stride = tpg / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      local_sum[tid] += local_sum[tid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write final sum
  if (tid == 0) {
    output[0] = local_sum[0];
  }
}

/**
 * @brief 平均値カーネル (sum を size で割る)
 *
 * このカーネルは sum の結果を要素数で割り、平均値を計算します。
 * 単一スレッドで実行されます。
 *
 * @param sum 総和 [[buffer(0)]]
 * @param output 平均値 [[buffer(1)]]
 * @param size 要素数 [[buffer(2)]]
 * @param gid Thread position in grid
 */
kernel void mean_kernel(device const float* sum [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid == 0) {
    output[0] = sum[0] / static_cast<float>(size);
  }
}
