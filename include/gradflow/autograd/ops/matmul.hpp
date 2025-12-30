#pragma once

#include "../tensor.hpp"

#include <stdexcept>

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

    size_t M = a.shape()[0];  // rows of a
    size_t K = a.shape()[1];  // cols of a = rows of b
    size_t K_b = b.shape()[0];  // rows of b
    size_t N = b.shape()[1];  // cols of b

    // Check that inner dimensions match
    if (K != K_b) {
        throw std::invalid_argument(
            "matmul requires matching inner dimensions: a[M, K] @ b[K, N]");
    }

    // Create result tensor [M, N]
    Tensor<T> result(Shape({M, N}));

    // Naive matrix multiplication implementation
    // result[i, j] = sum_k(a[i, k] * b[k, j])
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            T sum = T(0);
            for (size_t k = 0; k < K; ++k) {
                sum += a[{i, k}] * b[{k, j}];
            }
            result[{i, j}] = sum;
        }
    }

    return result;
}

}  // namespace gradflow
