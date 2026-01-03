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
 * @brief Square root operation with automatic differentiation
 *
 * Computes element-wise square root.
 *
 * Forward:
 *   z = sqrt(x)
 *
 * Backward:
 *   ∂L/∂x = ∂L/∂z / (2 * z)
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class SqrtOperation : public Operation<T> {
 public:
  Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
    if (inputs.size() != 1) {
      throw std::invalid_argument("SqrtOperation requires exactly 1 input");
    }

    auto result = sqrt(inputs[0]);

    // Save output for backward
    this->saveForBackward("output", result);

    return result;
  }

  std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
    auto output = this->getSavedTensor("output");

    // grad_x = grad_output / (2 * output)
    // Create a tensor filled with 2.0
    Tensor<T> two(output.shape());
    for (size_t i = 0; i < two.size(); ++i) {
      two.data()[i] = static_cast<T>(2);
    }

    auto denominator = mul(two, output);
    auto grad_x = div(grad_output, denominator);

    return {grad_x};
  }

  [[nodiscard]] std::string name() const override { return "SqrtOperation"; }
};

}  // namespace gradflow
