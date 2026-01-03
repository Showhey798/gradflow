#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../operation.hpp"
#include "../tensor.hpp"
#include "elementwise.hpp"

namespace gradflow {

/**
 * @brief Sum reduction over all elements
 *
 * @tparam T Element type
 * @param a Input tensor
 * @return Scalar tensor containing the sum
 */
template <typename T>
Tensor<T> sum(const Tensor<T>& a) {
  Tensor<T> result(Shape({}));  // Scalar
  T sum_value = T(0);

  detail::iterateIndices(a.shape(), [&](const std::vector<size_t>& indices) {
    sum_value += a[indices];
  });

  result[{}] = sum_value;
  return result;
}

/**
 * @brief Sum reduction along a specific axis
 *
 * @tparam T Element type
 * @param a Input tensor
 * @param axis Axis to reduce (0 to ndim-1)
 * @param keepdim If true, keep the reduced dimension with size 1
 * @return Tensor with reduced dimension
 * @throws std::out_of_range if axis is out of bounds
 */
template <typename T>
Tensor<T> sum(const Tensor<T>& a, size_t axis, bool keepdim = false) {
  if (axis >= a.ndim()) {
    throw std::out_of_range("Axis out of range");
  }

  // Compute output shape (remove the reduced axis or set to 1)
  std::vector<size_t> result_dims;
  for (size_t i = 0; i < a.ndim(); ++i) {
    if (i == axis) {
      if (keepdim) {
        result_dims.push_back(1);
      }
    } else {
      result_dims.push_back(a.shape()[i]);
    }
  }

  Shape result_shape(result_dims.empty() ? std::vector<size_t>{} : result_dims);
  Tensor<T> result(result_shape);

  // Initialize result to zero
  detail::iterateIndices(result_shape, [&](const std::vector<size_t>& indices) {
    result[indices] = T(0);
  });

  // Sum over the specified axis
  detail::iterateIndices(a.shape(), [&](const std::vector<size_t>& a_indices) {
    // Create result indices by removing or keeping the axis dimension
    std::vector<size_t> result_indices;
    for (size_t i = 0; i < a.ndim(); ++i) {
      if (i == axis) {
        if (keepdim) {
          result_indices.push_back(0);
        }
      } else {
        result_indices.push_back(a_indices[i]);
      }
    }
    result[result_indices] += a[a_indices];
  });

  return result;
}

/**
 * @brief Mean reduction over all elements
 *
 * @tparam T Element type
 * @param a Input tensor
 * @return Scalar tensor containing the mean
 */
template <typename T>
Tensor<T> mean(const Tensor<T>& a) {
  Tensor<T> result = sum(a);
  result[{}] /= static_cast<T>(a.size());
  return result;
}

/**
 * @brief Mean reduction along a specific axis
 *
 * @tparam T Element type
 * @param a Input tensor
 * @param axis Axis to reduce
 * @return Tensor with reduced dimension containing means
 * @throws std::out_of_range if axis is out of bounds
 */
template <typename T>
Tensor<T> mean(const Tensor<T>& a, size_t axis) {
  if (axis >= a.ndim()) {
    throw std::out_of_range("Axis out of range");
  }

  Tensor<T> result = sum(a, axis);
  T divisor = static_cast<T>(a.shape()[axis]);

  detail::iterateIndices(
      result.shape(),
      [&](const std::vector<size_t>& indices) { result[indices] /= divisor; });

  return result;
}

/**
 * @brief Max reduction over all elements
 *
 * @tparam T Element type
 * @param a Input tensor
 * @return Scalar tensor containing the maximum value
 */
template <typename T>
Tensor<T> max(const Tensor<T>& a) {
  if (a.size() == 0) {
    throw std::invalid_argument("Cannot compute max of empty tensor");
  }

  Tensor<T> result(Shape({}));
  T max_value = std::numeric_limits<T>::lowest();

  detail::iterateIndices(a.shape(), [&](const std::vector<size_t>& indices) {
    max_value = std::max(max_value, a[indices]);
  });

  result[{}] = max_value;
  return result;
}

/**
 * @brief Max reduction along a specific axis
 *
 * @tparam T Element type
 * @param a Input tensor
 * @param axis Axis to reduce
 * @param keepdim If true, keep the reduced dimension with size 1
 * @return Tensor with reduced dimension containing maximum values
 * @throws std::out_of_range if axis is out of bounds
 */
template <typename T>
Tensor<T> max(const Tensor<T>& a, size_t axis, bool keepdim = false) {
  if (axis >= a.ndim()) {
    throw std::out_of_range("Axis out of range");
  }

  // Compute output shape
  std::vector<size_t> result_dims;
  for (size_t i = 0; i < a.ndim(); ++i) {
    if (i == axis) {
      if (keepdim) {
        result_dims.push_back(1);
      }
    } else {
      result_dims.push_back(a.shape()[i]);
    }
  }

  Shape result_shape(result_dims.empty() ? std::vector<size_t>{} : result_dims);
  Tensor<T> result(result_shape);

  // Initialize result to lowest value
  detail::iterateIndices(result_shape, [&](const std::vector<size_t>& indices) {
    result[indices] = std::numeric_limits<T>::lowest();
  });

  // Find max over the specified axis
  detail::iterateIndices(a.shape(), [&](const std::vector<size_t>& a_indices) {
    std::vector<size_t> result_indices;
    for (size_t i = 0; i < a.ndim(); ++i) {
      if (i == axis) {
        if (keepdim) {
          result_indices.push_back(0);
        }
      } else {
        result_indices.push_back(a_indices[i]);
      }
    }
    result[result_indices] = std::max(result[result_indices], a[a_indices]);
  });

  return result;
}

/**
 * @brief Min reduction over all elements
 *
 * @tparam T Element type
 * @param a Input tensor
 * @return Scalar tensor containing the minimum value
 */
template <typename T>
Tensor<T> min(const Tensor<T>& a) {
  if (a.size() == 0) {
    throw std::invalid_argument("Cannot compute min of empty tensor");
  }

  Tensor<T> result(Shape({}));
  T min_value = std::numeric_limits<T>::max();

  detail::iterateIndices(a.shape(), [&](const std::vector<size_t>& indices) {
    min_value = std::min(min_value, a[indices]);
  });

  result[{}] = min_value;
  return result;
}

/**
 * @brief Min reduction along a specific axis
 *
 * @tparam T Element type
 * @param a Input tensor
 * @param axis Axis to reduce
 * @return Tensor with reduced dimension containing minimum values
 * @throws std::out_of_range if axis is out of bounds
 */
template <typename T>
Tensor<T> min(const Tensor<T>& a, size_t axis) {
  if (axis >= a.ndim()) {
    throw std::out_of_range("Axis out of range");
  }

  // Compute output shape
  std::vector<size_t> result_dims;
  for (size_t i = 0; i < a.ndim(); ++i) {
    if (i != axis) {
      result_dims.push_back(a.shape()[i]);
    }
  }

  Shape result_shape(result_dims.empty() ? std::vector<size_t>{} : result_dims);
  Tensor<T> result(result_shape);

  // Initialize result to max value
  detail::iterateIndices(result_shape, [&](const std::vector<size_t>& indices) {
    result[indices] = std::numeric_limits<T>::max();
  });

  // Find min over the specified axis
  detail::iterateIndices(a.shape(), [&](const std::vector<size_t>& a_indices) {
    std::vector<size_t> result_indices;
    for (size_t i = 0; i < a.ndim(); ++i) {
      if (i != axis) {
        result_indices.push_back(a_indices[i]);
      }
    }
    result[result_indices] = std::min(result[result_indices], a[a_indices]);
  });

  return result;
}

// ========================================
// Reduction Operations for Autograd
// ========================================

/**
 * @brief Sum reduction operation with automatic differentiation
 *
 * Computes the sum of all elements in the input tensor.
 *
 * Forward:
 *   y = Σ(x)
 *
 * Backward:
 *   ∂L/∂x = ∂L/∂y * 1 (broadcasted to input shape)
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class SumOperation : public Operation<T> {
 public:
  /**
   * @brief Forward pass: compute sum
   *
   * @param inputs Vector containing exactly 1 tensor
   * @return Scalar tensor containing the sum
   * @throws std::invalid_argument if inputs size is not 1
   */
  Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
    if (inputs.size() != 1) {
      throw std::invalid_argument("SumOperation requires exactly 1 input");
    }

    const auto& x = inputs[0];

    // Save input shape for backward
    this->saveForBackward("input_shape_holder", Tensor<T>(x.shape()));

    return sum(x);
  }

  /**
   * @brief Backward pass: compute gradients
   *
   * @param grad_output Gradient of loss (scalar)
   * @return Vector of gradients [grad_x]
   */
  std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
    auto input_shape = this->getSavedTensor("input_shape_holder").shape();

    // Gradient is broadcasted to input shape
    // grad_x = grad_output * 1 (for each element)
    Tensor<T> grad_x(input_shape);
    T grad_value = grad_output[{}];  // Scalar value

    for (size_t i = 0; i < grad_x.size(); ++i) {
      grad_x.data()[i] = grad_value;
    }

    return {grad_x};
  }

  [[nodiscard]] std::string name() const override { return "SumOperation"; }
};

}  // namespace gradflow
