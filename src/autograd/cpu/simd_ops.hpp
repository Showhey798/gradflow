// src/autograd/cpu/simd_ops.hpp
#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

// x86_64 アーキテクチャでのみ AVX2 intrinsics をインクルード
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>  // AVX2/AVX512 intrinsics
#endif

namespace gradflow {
namespace cpu {
namespace simd {

#if defined(__x86_64__) || defined(_M_X64)
/**
 * @brief AVX2 を使用した要素ごとの加算
 *
 * 32 バイトアライメント必須。サイズは 8 の倍数であることを推奨。
 *
 * @param a 入力配列 A (32-byte aligned)
 * @param b 入力配列 B (32-byte aligned)
 * @param c 出力配列 C (32-byte aligned)
 * @param size 要素数
 */
inline void add_avx2(const float* a, const float* b, float* c, size_t size) {
#ifndef NDEBUG
  assert(reinterpret_cast<uintptr_t>(a) % 32 == 0 &&
         "a must be 32-byte aligned");
  assert(reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
         "b must be 32-byte aligned");
  assert(reinterpret_cast<uintptr_t>(c) % 32 == 0 &&
         "c must be 32-byte aligned");
#endif

  size_t i = 0;

  // AVX2: 8 個の float を同時処理
  for (; i + 8 <= size; i += 8) {
    __m256 va = _mm256_load_ps(a + i);  // Aligned load
    __m256 vb = _mm256_load_ps(b + i);
    __m256 vc = _mm256_add_ps(va, vb);
    _mm256_store_ps(c + i, vc);  // Aligned store
  }

  // 残りの要素をスカラー処理
  for (; i < size; ++i) {
    c[i] = a[i] + b[i];
  }
}
#endif

#if defined(__x86_64__) || defined(_M_X64)
/**
 * @brief AVX2 を使用した要素ごとの乗算
 *
 * @param a 入力配列 A (32-byte aligned)
 * @param b 入力配列 B (32-byte aligned)
 * @param c 出力配列 C (32-byte aligned)
 * @param size 要素数
 */
inline void mul_avx2(const float* a, const float* b, float* c, size_t size) {
#ifndef NDEBUG
  assert(reinterpret_cast<uintptr_t>(a) % 32 == 0 &&
         "a must be 32-byte aligned");
  assert(reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
         "b must be 32-byte aligned");
  assert(reinterpret_cast<uintptr_t>(c) % 32 == 0 &&
         "c must be 32-byte aligned");
#endif

  size_t i = 0;

  for (; i + 8 <= size; i += 8) {
    __m256 va = _mm256_load_ps(a + i);
    __m256 vb = _mm256_load_ps(b + i);
    __m256 vc = _mm256_mul_ps(va, vb);
    _mm256_store_ps(c + i, vc);
  }

  for (; i < size; ++i) {
    c[i] = a[i] * b[i];
  }
}

/**
 * @brief AVX2 を使用した要素ごとの減算
 *
 * @param a 入力配列 A (32-byte aligned)
 * @param b 入力配列 B (32-byte aligned)
 * @param c 出力配列 C (32-byte aligned)
 * @param size 要素数
 */
inline void sub_avx2(const float* a, const float* b, float* c, size_t size) {
#ifndef NDEBUG
  assert(reinterpret_cast<uintptr_t>(a) % 32 == 0 &&
         "a must be 32-byte aligned");
  assert(reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
         "b must be 32-byte aligned");
  assert(reinterpret_cast<uintptr_t>(c) % 32 == 0 &&
         "c must be 32-byte aligned");
#endif

  size_t i = 0;

  for (; i + 8 <= size; i += 8) {
    __m256 va = _mm256_load_ps(a + i);
    __m256 vb = _mm256_load_ps(b + i);
    __m256 vc = _mm256_sub_ps(va, vb);
    _mm256_store_ps(c + i, vc);
  }

  for (; i < size; ++i) {
    c[i] = a[i] - b[i];
  }
}

/**
 * @brief AVX2 を使用した要素ごとの除算
 *
 * @param a 入力配列 A (32-byte aligned)
 * @param b 入力配列 B (32-byte aligned)
 * @param c 出力配列 C (32-byte aligned)
 * @param size 要素数
 */
inline void div_avx2(const float* a, const float* b, float* c, size_t size) {
#ifndef NDEBUG
  assert(reinterpret_cast<uintptr_t>(a) % 32 == 0 &&
         "a must be 32-byte aligned");
  assert(reinterpret_cast<uintptr_t>(b) % 32 == 0 &&
         "b must be 32-byte aligned");
  assert(reinterpret_cast<uintptr_t>(c) % 32 == 0 &&
         "c must be 32-byte aligned");
#endif

  size_t i = 0;

  for (; i + 8 <= size; i += 8) {
    __m256 va = _mm256_load_ps(a + i);
    __m256 vb = _mm256_load_ps(b + i);
    __m256 vc = _mm256_div_ps(va, vb);
    _mm256_store_ps(c + i, vc);
  }

  for (; i < size; ++i) {
    c[i] = a[i] / b[i];
  }
}
#endif

}  // namespace simd
}  // namespace cpu
}  // namespace gradflow
