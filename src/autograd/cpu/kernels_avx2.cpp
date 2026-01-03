// src/autograd/cpu/kernels_avx2.cpp
#include "gradflow/autograd/cpu/kernels.hpp"
#include "simd_ops.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

#include <cstdlib>
#include <cstring>

namespace gradflow {
namespace cpu {

// CPUID による機能検出
static bool has_avx2() {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
  int cpuInfo[4];
  __cpuidex(cpuInfo, 7, 0);
  return (cpuInfo[1] & (1 << 5)) != 0;  // AVX2 bit in EBX
#else
  unsigned int eax, ebx, ecx, edx;
  if (__get_cpuid(7, &eax, &ebx, &ecx, &edx)) {
    return (ebx & (1 << 5)) != 0;  // AVX2 bit
  }
#endif
#endif
  return false;
}

static bool has_avx512() {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
  int cpuInfo[4];
  __cpuidex(cpuInfo, 7, 0);
  return (cpuInfo[1] & (1 << 16)) != 0;  // AVX-512F bit in EBX
#else
  unsigned int eax, ebx, ecx, edx;
  if (__get_cpuid(7, &eax, &ebx, &ecx, &edx)) {
    return (ebx & (1 << 16)) != 0;  // AVX-512F bit
  }
#endif
#endif
  return false;
}

CPUKernels::CPUKernels() : has_avx2_(has_avx2()), has_avx512_(has_avx512()) {}

CPUKernels::~CPUKernels() = default;

void CPUKernels::add(const float* a, const float* b, float* c, size_t size) {
  // 入力検証
  if (a == nullptr || b == nullptr || c == nullptr) {
    return;  // nullptr の場合は何もしない
  }
  if (size == 0) {
    return;  // サイズが 0 の場合は何もしない
  }

#if defined(__x86_64__) || defined(_M_X64)
  if (has_avx2_) {
    simd::add_avx2(a, b, c, size);
  } else
#endif
  {
    // Fallback: スカラー実装
    for (size_t i = 0; i < size; ++i) {
      c[i] = a[i] + b[i];
    }
  }
}

void CPUKernels::mul(const float* a, const float* b, float* c, size_t size) {
  // 入力検証
  if (a == nullptr || b == nullptr || c == nullptr) {
    return;
  }
  if (size == 0) {
    return;
  }

#if defined(__x86_64__) || defined(_M_X64)
  if (has_avx2_) {
    simd::mul_avx2(a, b, c, size);
  } else
#endif
  {
    for (size_t i = 0; i < size; ++i) {
      c[i] = a[i] * b[i];
    }
  }
}

void CPUKernels::sub(const float* a, const float* b, float* c, size_t size) {
  // 入力検証
  if (a == nullptr || b == nullptr || c == nullptr) {
    return;
  }
  if (size == 0) {
    return;
  }

#if defined(__x86_64__) || defined(_M_X64)
  if (has_avx2_) {
    simd::sub_avx2(a, b, c, size);
  } else
#endif
  {
    for (size_t i = 0; i < size; ++i) {
      c[i] = a[i] - b[i];
    }
  }
}

void CPUKernels::div(const float* a, const float* b, float* c, size_t size) {
  // 入力検証
  if (a == nullptr || b == nullptr || c == nullptr) {
    return;
  }
  if (size == 0) {
    return;
  }

#if defined(__x86_64__) || defined(_M_X64)
  if (has_avx2_) {
    simd::div_avx2(a, b, c, size);
  } else
#endif
  {
    for (size_t i = 0; i < size; ++i) {
      c[i] = a[i] / b[i];
    }
  }
}

std::string CPUKernels::getSIMDInfo() const {
  // 注: AVX-512 実装は未完成のため、AVX2 を返す
  // TODO: AVX-512 実装が完成したら has_avx512_ チェックを有効化
  if (has_avx2_) {
    return "AVX2";
  } else {
    return "Scalar (No SIMD)";
  }
}

void* alignedAlloc(size_t size, size_t alignment) {
#if defined(_MSC_VER)
  return _aligned_malloc(size, alignment);
#else
  void* ptr = nullptr;
  if (posix_memalign(&ptr, alignment, size) != 0) {
    return nullptr;
  }
  return ptr;
#endif
}

void alignedFree(void* ptr) {
#if defined(_MSC_VER)
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

}  // namespace cpu
}  // namespace gradflow
