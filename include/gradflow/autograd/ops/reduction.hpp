#pragma once

#include "../tensor.hpp"
#include "elementwise.hpp"

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>

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

    detail::iterate_indices(a.shape(),
                            [&](const std::vector<size_t>& indices) { sum_value += a[indices]; });

    result[{}] = sum_value;
    return result;
}

/**
 * @brief Sum reduction along a specific axis
 *
 * @tparam T Element type
 * @param a Input tensor
 * @param axis Axis to reduce (0 to ndim-1)
 * @return Tensor with reduced dimension
 * @throws std::out_of_range if axis is out of bounds
 */
template <typename T>
Tensor<T> sum(const Tensor<T>& a, size_t axis) {
    if (axis >= a.ndim()) {
        throw std::out_of_range("Axis out of range");
    }

    // Compute output shape (remove the reduced axis)
    std::vector<size_t> result_dims;
    for (size_t i = 0; i < a.ndim(); ++i) {
        if (i != axis) {
            result_dims.push_back(a.shape()[i]);
        }
    }

    Shape result_shape(result_dims.empty() ? std::vector<size_t>{} : result_dims);
    Tensor<T> result(result_shape);

    // Initialize result to zero
    detail::iterate_indices(result_shape,
                            [&](const std::vector<size_t>& indices) { result[indices] = T(0); });

    // Sum over the specified axis
    detail::iterate_indices(a.shape(), [&](const std::vector<size_t>& a_indices) {
        // Create result indices by removing the axis dimension
        std::vector<size_t> result_indices;
        for (size_t i = 0; i < a.ndim(); ++i) {
            if (i != axis) {
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

    detail::iterate_indices(
        result.shape(), [&](const std::vector<size_t>& indices) { result[indices] /= divisor; });

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

    detail::iterate_indices(a.shape(), [&](const std::vector<size_t>& indices) {
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
 * @return Tensor with reduced dimension containing maximum values
 * @throws std::out_of_range if axis is out of bounds
 */
template <typename T>
Tensor<T> max(const Tensor<T>& a, size_t axis) {
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

    // Initialize result to lowest value
    detail::iterate_indices(result_shape, [&](const std::vector<size_t>& indices) {
        result[indices] = std::numeric_limits<T>::lowest();
    });

    // Find max over the specified axis
    detail::iterate_indices(a.shape(), [&](const std::vector<size_t>& a_indices) {
        std::vector<size_t> result_indices;
        for (size_t i = 0; i < a.ndim(); ++i) {
            if (i != axis) {
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

    detail::iterate_indices(a.shape(), [&](const std::vector<size_t>& indices) {
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
    detail::iterate_indices(result_shape, [&](const std::vector<size_t>& indices) {
        result[indices] = std::numeric_limits<T>::max();
    });

    // Find min over the specified axis
    detail::iterate_indices(a.shape(), [&](const std::vector<size_t>& a_indices) {
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

}  // namespace gradflow
