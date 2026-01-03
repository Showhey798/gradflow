#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../operation.hpp"
#include "../tensor.hpp"
#include "elementwise.hpp"

namespace gradflow {

/**
 * @brief ReLU (Rectified Linear Unit) operation
 *
 * Applies the rectified linear unit function element-wise.
 *
 * Forward:
 *   y = max(0, x)
 *
 * Backward:
 *   ∂L/∂x = ∂L/∂y * (x > 0 ? 1 : 0)
 *
 * Properties:
 * - Non-saturating activation function
 * - Widely used in deep learning
 * - Simple and fast to compute
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class ReLUOperation : public Operation<T> {
 public:
  /**
   * @brief Forward pass: apply ReLU function
   *
   * @param inputs Vector of input tensors (size must be 1)
   * @return Output tensor with ReLU applied
   * @throws std::invalid_argument if inputs size is not 1
   */
  Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
    if (inputs.size() != 1) {
      throw std::invalid_argument("ReLUOperation requires exactly 1 input");
    }

    const auto& x = inputs[0];

    // y = max(0, x)
    // Implement directly for consistency with LeakyReLU
    Tensor<T> result(x.shape());
    for (size_t i = 0; i < x.size(); ++i) {
      result.data()[i] = x.data()[i] > T(0) ? x.data()[i] : T(0);
    }

    // Save input for backward
    this->saveForBackward("input", x);

    return result;
  }

  /**
   * @brief Backward pass: compute gradient
   *
   * @param grad_output Gradient of loss with respect to output
   * @return Vector of gradients with respect to input
   */
  std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
    auto input = this->getSavedTensor("input");

    // grad_x = grad_output * (x > 0 ? 1 : 0)
    // Create mask: 1 where input > 0, 0 otherwise
    Tensor<T> mask(input.shape());
    for (size_t i = 0; i < input.size(); ++i) {
      mask.data()[i] = input.data()[i] > T(0) ? T(1) : T(0);
    }

    auto grad_x = mul(grad_output, mask);

    return {grad_x};
  }

  [[nodiscard]] std::string name() const override { return "ReLUOperation"; }
};

}  // namespace gradflow
