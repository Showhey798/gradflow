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
 * @brief Tanh (Hyperbolic Tangent) activation function
 *
 * Applies the hyperbolic tangent function element-wise.
 *
 * Forward:
 *   y = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 *
 * Backward:
 *   ∂L/∂x = ∂L/∂y * (1 - y²)
 *
 * Properties:
 * - Output range: [-1, 1]
 * - Zero-centered (unlike sigmoid)
 * - Can suffer from vanishing gradients for large |x|
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class TanhOperation : public Operation<T> {
 public:
  Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
    if (inputs.size() != 1) {
      throw std::invalid_argument("TanhOperation requires exactly 1 input");
    }

    const auto& x = inputs[0];

    // Use Tensor-level tanh
    auto result = tanh(x);

    // Save output for backward
    this->saveForBackward("output", result);

    return result;
  }

  std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
    auto output = this->getSavedTensor("output");

    // grad_x = grad_output * (1 - output²)
    auto output_squared = mul(output, output);

    Tensor<T> one_minus_output_squared(output_squared.shape());
    for (size_t i = 0; i < output_squared.size(); ++i) {
      one_minus_output_squared.data()[i] = T(1) - output_squared.data()[i];
    }

    auto grad_x = mul(grad_output, one_minus_output_squared);

    return {grad_x};
  }

  [[nodiscard]] std::string name() const override { return "TanhOperation"; }
};

}  // namespace gradflow
