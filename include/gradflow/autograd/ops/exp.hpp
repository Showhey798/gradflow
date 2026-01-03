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
 * @brief Exponential operation with automatic differentiation
 *
 * Computes element-wise exponential function.
 *
 * Forward:
 *   z = exp(x)
 *
 * Backward:
 *   ∂L/∂x = ∂L/∂z * z  (since dexp(x)/dx = exp(x) = z)
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class ExpOperation : public Operation<T> {
 public:
  Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
    if (inputs.size() != 1) {
      throw std::invalid_argument("ExpOperation requires exactly 1 input");
    }

    auto result = exp(inputs[0]);

    // Save output for backward (more efficient than recomputing exp)
    this->saveForBackward("output", result);

    return result;
  }

  std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
    auto output = this->getSavedTensor("output");

    // grad_x = grad_output * output (since output = exp(x))
    auto grad_x = mul(grad_output, output);

    return {grad_x};
  }

  [[nodiscard]] std::string name() const override { return "ExpOperation"; }
};

}  // namespace gradflow
