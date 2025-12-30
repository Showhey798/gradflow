#pragma once

#include "../operation.hpp"
#include "../tensor.hpp"
#include "reduction.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace gradflow {
namespace ops {

/**
 * @brief Adjust gradient for broadcasting
 *
 * When broadcasting happens during forward pass, the gradient needs to be
 * summed over the broadcasted dimensions during backward pass.
 *
 * Example:
 *   Forward: a[3, 1] + b[3, 4] -> c[3, 4]  (a is broadcasted along dim 1)
 *   Backward: grad_a needs to be summed over dimension 1: [3, 4] -> [3, 1]
 *
 * @tparam T Element type
 * @param grad Gradient tensor (same shape as forward output)
 * @param target_shape Target shape (shape of the input during forward)
 * @return Adjusted gradient tensor (same shape as target_shape)
 */
template <typename T>
Tensor<T> sumToShape(const Tensor<T>& grad, const Shape& target_shape) {
    // If shapes are already the same, return as is
    if (grad.shape() == target_shape) {
        return grad;
    }

    Tensor<T> result = grad;

    // Step 1: Sum over prepended dimensions (when grad has more dims than target)
    while (result.ndim() > target_shape.ndim()) {
        result = sum(result, 0);
    }

    // Step 2: Sum over dimensions that were broadcasted (where target dim is 1)
    // We iterate in reverse order to avoid index shifting issues
    for (size_t i = target_shape.ndim(); i > 0; --i) {
        size_t dim_idx = i - 1;
        if (target_shape[dim_idx] == 1 && result.shape()[dim_idx] != 1) {
            result = sum(result, dim_idx);
            // After sum, the dimension is removed. We need to reshape to add it back.
            std::vector<size_t> new_shape_vec;
            for (size_t j = 0; j < target_shape.ndim(); ++j) {
                if (j == dim_idx) {
                    new_shape_vec.push_back(1);
                } else if (j < dim_idx) {
                    new_shape_vec.push_back(result.shape()[j]);
                } else {
                    new_shape_vec.push_back(result.shape()[j - 1]);
                }
            }
            result = result.reshape(Shape(new_shape_vec));
        }
    }

    return result;
}

namespace test {

/**
 * @brief Numerical gradient checker
 *
 * Compares automatic differentiation gradient with numerical gradient
 * using finite difference method: f'(x) ≈ (f(x + ε) - f(x - ε)) / (2ε)
 *
 * @tparam T Element type
 * @param op Operation to test
 * @param inputs Input tensors
 * @param output_index Which output element to use for gradient check (default: first element)
 * @param epsilon Finite difference step size (default: 1e-4)
 * @param tolerance Acceptable relative error (default: 1e-2)
 * @return True if all gradients are within tolerance
 */
template <typename T>
bool checkNumericalGradient(Operation<T>& op,
                            const std::vector<Tensor<T>>& inputs,
                            const std::vector<size_t>& output_index = {},
                            T epsilon = static_cast<T>(1e-4),
                            T tolerance = static_cast<T>(1e-2)) {
    // Forward pass to get output and compute analytical gradients
    auto output = op.forward(inputs);

    // Create grad_output (all zeros except 1 at output_index)
    Tensor<T> grad_output(output.shape());
    // Initialize to zero
    for (size_t i = 0; i < grad_output.size(); ++i) {
        grad_output.data()[i] = T(0);
    }
    // Set 1 at the target index
    if (output_index.empty()) {
        grad_output[std::vector<size_t>(output.ndim(), 0)] = T(1);
    } else {
        grad_output[output_index] = T(1);
    }

    // Backward pass to get analytical gradients
    auto analytical_grads = op.backward(grad_output);

    if (analytical_grads.size() != inputs.size()) {
        return false;
    }

    // Check gradient for each input
    for (size_t input_idx = 0; input_idx < inputs.size(); ++input_idx) {
        const auto& input = inputs[input_idx];
        const auto& analytical_grad = analytical_grads[input_idx];

        if (analytical_grad.shape() != input.shape()) {
            return false;
        }

        // Check gradient for each element of the input
        for (size_t i = 0; i < input.size(); ++i) {
            // Create copies of inputs for perturbation
            std::vector<Tensor<T>> inputs_plus;
            std::vector<Tensor<T>> inputs_minus;

            // Deep copy inputs
            for (const auto& inp : inputs) {
                inputs_plus.push_back(Tensor<T>(inp.shape()));
                inputs_minus.push_back(Tensor<T>(inp.shape()));
                for (size_t j = 0; j < inp.size(); ++j) {
                    inputs_plus.back().data()[j] = inp.data()[j];
                    inputs_minus.back().data()[j] = inp.data()[j];
                }
            }

            // Perturb the i-th element
            inputs_plus[input_idx].data()[i] += epsilon;
            inputs_minus[input_idx].data()[i] -= epsilon;

            // Forward pass with perturbed inputs (using fresh operation state)
            auto output_plus = op.forward(inputs_plus);
            auto output_minus = op.forward(inputs_minus);

            // Numerical gradient: (f(x + ε) - f(x - ε)) / (2ε)
            T output_val_plus;
            T output_val_minus;

            if (output_index.empty()) {
                output_val_plus = output_plus[std::vector<size_t>(output_plus.ndim(), 0)];
                output_val_minus = output_minus[std::vector<size_t>(output_minus.ndim(), 0)];
            } else {
                output_val_plus = output_plus[output_index];
                output_val_minus = output_minus[output_index];
            }

            T numerical_grad = (output_val_plus - output_val_minus) / (T(2) * epsilon);

            // Compare with analytical gradient
            T analytical_grad_val = analytical_grad.data()[i];

            // Compute relative error
            T abs_diff = std::abs(numerical_grad - analytical_grad_val);
            T max_val = std::max(std::abs(numerical_grad), std::abs(analytical_grad_val));
            T relative_error = (max_val < T(1e-7)) ? abs_diff : (abs_diff / max_val);

            if (relative_error > tolerance) {
                return false;
            }
        }
    }

    return true;
}

}  // namespace test
}  // namespace ops
}  // namespace gradflow
