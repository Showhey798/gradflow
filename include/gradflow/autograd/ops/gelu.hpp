#pragma once

#include "../operation.hpp"
#include "../tensor.hpp"
#include "elementwise.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gradflow {

/**
 * @brief GELU (Gaussian Error Linear Unit) activation function
 *
 * Applies the Gaussian Error Linear Unit function element-wise.
 * This implementation uses the tanh approximation for efficiency.
 *
 * Forward (approximate):
 *   y = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 *
 * Backward:
 *   ∂L/∂x = ∂L/∂y * ∂y/∂x
 *   where ∂y/∂x is computed using the chain rule
 *
 * Properties:
 * - Smooth activation function
 * - Recommended for transformer architectures
 * - Provides better gradients than ReLU in some cases
 *
 * References:
 * - Original paper: "Gaussian Error Linear Units (GELUs)"
 * - PyTorch implementation: torch.nn.GELU(approximate='tanh')
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class GELUOperation : public Operation<T> {
public:
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("GELUOperation requires exactly 1 input");
        }

        const auto& x = inputs[0];

        // Constants
        constexpr T kSqrt2OverPi = static_cast<T>(0.7978845608028654);  // √(2/π)
        constexpr T kCoeff = static_cast<T>(0.044715);

        // Compute: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        // Optimize by reducing intermediate tensor allocations
        Tensor<T> scaled_inner(x.shape());
        for (size_t i = 0; i < x.size(); ++i) {
            T x_val = x.data()[i];
            T x_cubed = x_val * x_val * x_val;
            scaled_inner.data()[i] = kSqrt2OverPi * (x_val + kCoeff * x_cubed);
        }

        auto tanh_value = tanh(scaled_inner);

        Tensor<T> result(x.shape());
        for (size_t i = 0; i < x.size(); ++i) {
            result.data()[i] = T(0.5) * x.data()[i] * (T(1) + tanh_value.data()[i]);
        }

        // Save for backward
        this->saveForBackward("input", x);
        this->saveForBackward("tanh_value", tanh_value);

        return result;
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto input = this->getSavedTensor("input");
        auto tanh_value = this->getSavedTensor("tanh_value");

        // Constants
        constexpr T kSqrt2OverPi = static_cast<T>(0.7978845608028654);  // √(2/π)
        constexpr T kCoeff = static_cast<T>(0.044715);

        // Compute ∂y/∂x
        // cdf = 0.5 * (1 + tanh(...))
        Tensor<T> cdf(tanh_value.shape());
        for (size_t i = 0; i < tanh_value.size(); ++i) {
            cdf.data()[i] = T(0.5) * (T(1) + tanh_value.data()[i]);
        }

        // pdf_approximation = (1 - tanh²) * √(2/π) * (1 + 3 * 0.044715 * x²)
        Tensor<T> tanh_squared(tanh_value.shape());
        for (size_t i = 0; i < tanh_value.size(); ++i) {
            tanh_squared.data()[i] = tanh_value.data()[i] * tanh_value.data()[i];
        }

        Tensor<T> one_minus_tanh_squared(tanh_squared.shape());
        for (size_t i = 0; i < tanh_squared.size(); ++i) {
            one_minus_tanh_squared.data()[i] = T(1) - tanh_squared.data()[i];
        }

        Tensor<T> x_squared(input.shape());
        for (size_t i = 0; i < input.size(); ++i) {
            x_squared.data()[i] = input.data()[i] * input.data()[i];
        }

        T three_coeff = T(3) * kCoeff;
        Tensor<T> three_coeff_x_squared(x_squared.shape());
        for (size_t i = 0; i < x_squared.size(); ++i) {
            three_coeff_x_squared.data()[i] = three_coeff * x_squared.data()[i];
        }

        Tensor<T> one_plus_three_coeff_x_squared(three_coeff_x_squared.shape());
        for (size_t i = 0; i < three_coeff_x_squared.size(); ++i) {
            one_plus_three_coeff_x_squared.data()[i] = T(1) + three_coeff_x_squared.data()[i];
        }

        Tensor<T> pdf_part(one_minus_tanh_squared.shape());
        for (size_t i = 0; i < one_minus_tanh_squared.size(); ++i) {
            pdf_part.data()[i] = one_minus_tanh_squared.data()[i] * kSqrt2OverPi;
        }

        Tensor<T> pdf_approximation(pdf_part.shape());
        for (size_t i = 0; i < pdf_part.size(); ++i) {
            pdf_approximation.data()[i] =
                pdf_part.data()[i] * one_plus_three_coeff_x_squared.data()[i];
        }

        // ∂y/∂x = cdf + 0.5 * x * pdf_approximation
        Tensor<T> half_x_pdf(input.shape());
        for (size_t i = 0; i < input.size(); ++i) {
            half_x_pdf.data()[i] = T(0.5) * input.data()[i] * pdf_approximation.data()[i];
        }

        Tensor<T> dy_dx(cdf.shape());
        for (size_t i = 0; i < cdf.size(); ++i) {
            dy_dx.data()[i] = cdf.data()[i] + half_x_pdf.data()[i];
        }

        // grad_x = grad_output * dy_dx
        auto grad_x = mul(grad_output, dy_dx);

        return {grad_x};
    }

    [[nodiscard]] std::string name() const override { return "GELUOperation"; }
};

}  // namespace gradflow
