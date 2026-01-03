#pragma once

#include <stdexcept>

#include "../tensor.hpp"

namespace gradflow {

/**
 * @brief Matrix multiplication operation
 *
 * Performs matrix multiplication of two 2D tensors.
 * For tensors a[M, K] and b[K, N], produces result[M, N].
 *
 * @tparam T Element type
 * @param a First tensor (M x K)
 * @param b Second tensor (K x N)
 * @return Result tensor (M x N)
 * @throws std::invalid_argument if tensors are not 2D or dimensions don't match
 */
template <typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
  // Check that both tensors are 2D
  if (a.ndim() != 2 || b.ndim() != 2) {
    throw std::invalid_argument("matmul requires 2D tensors");
  }

  const size_t kM = a.shape()[0];   // rows of a
  const size_t kK = a.shape()[1];   // cols of a = rows of b
  const size_t kKB = b.shape()[0];  // rows of b
  const size_t kN = b.shape()[1];   // cols of b

  // Check that inner dimensions match
  if (kK != kKB) {
    throw std::invalid_argument(
        "matmul requires matching inner dimensions: a[m, k] @ b[k, n]");
  }

  // Create result tensor [m, n]
  Tensor<T> result(Shape({kM, kN}));

  // Naive matrix multiplication implementation
  // result[i, j] = sum_k(a[i, k] * b[k, j])
  for (size_t i = 0; i < kM; ++i) {
    for (size_t j = 0; j < kN; ++j) {
      T sum = T(0);
      for (size_t k_idx = 0; k_idx < kK; ++k_idx) {
        sum += a[{i, k_idx}] * b[{k_idx, j}];
      }
      result[{i, j}] = sum;
    }
  }

  return result;
}

}  // namespace gradflow
