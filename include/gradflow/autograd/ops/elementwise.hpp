#pragma once

#include "../tensor.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

namespace gradflow {

namespace detail {

/**
 * @brief Helper function to broadcast two shapes
 *
 * @param shape1 First shape
 * @param shape2 Second shape
 * @return Broadcast shape
 * @throws std::invalid_argument if shapes are not broadcastable
 */
inline Shape broadcast_shapes(const Shape& shape1, const Shape& shape2) {
    if (!shape1.is_broadcastable_with(shape2)) {
        throw std::invalid_argument("Shapes are not broadcastable");
    }
    return shape1.broadcast_with(shape2);
}

/**
 * @brief Helper function to get broadcasted indices
 *
 * Maps output indices to input indices considering broadcasting rules.
 *
 * @param output_indices Indices in the output tensor
 * @param input_shape Shape of the input tensor
 * @param output_shape Shape of the output tensor
 * @return Corresponding indices in the input tensor
 */
inline std::vector<size_t> get_broadcast_indices(const std::vector<size_t>& output_indices,
                                                  const Shape& input_shape,
                                                  const Shape& output_shape) {
    size_t input_ndim = input_shape.ndim();
    size_t output_ndim = output_shape.ndim();
    std::vector<size_t> input_indices(input_ndim);

    // Process dimensions from right to left
    for (size_t i = 0; i < output_ndim; ++i) {
        size_t output_idx = output_indices[output_ndim - 1 - i];

        if (i < input_ndim) {
            size_t input_dim = input_shape[input_ndim - 1 - i];
            // If input dimension is 1, broadcast (use index 0)
            input_indices[input_ndim - 1 - i] = (input_dim == 1) ? 0 : output_idx;
        }
        // If i >= input_ndim, the dimension doesn't exist in input (implicit 1)
    }

    return input_indices;
}

/**
 * @brief Helper function to iterate over all indices of a shape
 *
 * @param shape Shape to iterate over
 * @param func Function to call for each index
 */
template <typename Func>
void iterate_indices(const Shape& shape, Func func) {
    if (shape.ndim() == 0) {
        func(std::vector<size_t>{});
        return;
    }

    std::vector<size_t> indices(shape.ndim(), 0);

    // Helper lambda to recursively iterate through dimensions
    std::function<void(size_t)> iterate_dim = [&](size_t dim) {
        if (dim == shape.ndim()) {
            func(indices);
            return;
        }

        for (size_t i = 0; i < shape[dim]; ++i) {
            indices[dim] = i;
            iterate_dim(dim + 1);
        }
    };

    iterate_dim(0);
}

}  // namespace detail

/**
 * @brief Element-wise addition with broadcasting support
 *
 * @tparam T Element type
 * @param a First tensor
 * @param b Second tensor
 * @return Result tensor with broadcasted shape
 * @throws std::invalid_argument if shapes are not broadcastable
 */
template <typename T>
Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b) {
    Shape result_shape = detail::broadcast_shapes(a.shape(), b.shape());
    Tensor<T> result(result_shape);

    detail::iterate_indices(result_shape, [&](const std::vector<size_t>& indices) {
        auto a_indices = detail::get_broadcast_indices(indices, a.shape(), result_shape);
        auto b_indices = detail::get_broadcast_indices(indices, b.shape(), result_shape);
        result[indices] = a[a_indices] + b[b_indices];
    });

    return result;
}

/**
 * @brief Element-wise subtraction with broadcasting support
 *
 * @tparam T Element type
 * @param a First tensor (minuend)
 * @param b Second tensor (subtrahend)
 * @return Result tensor with broadcasted shape
 * @throws std::invalid_argument if shapes are not broadcastable
 */
template <typename T>
Tensor<T> sub(const Tensor<T>& a, const Tensor<T>& b) {
    Shape result_shape = detail::broadcast_shapes(a.shape(), b.shape());
    Tensor<T> result(result_shape);

    detail::iterate_indices(result_shape, [&](const std::vector<size_t>& indices) {
        auto a_indices = detail::get_broadcast_indices(indices, a.shape(), result_shape);
        auto b_indices = detail::get_broadcast_indices(indices, b.shape(), result_shape);
        result[indices] = a[a_indices] - b[b_indices];
    });

    return result;
}

/**
 * @brief Element-wise multiplication with broadcasting support
 *
 * @tparam T Element type
 * @param a First tensor
 * @param b Second tensor
 * @return Result tensor with broadcasted shape
 * @throws std::invalid_argument if shapes are not broadcastable
 */
template <typename T>
Tensor<T> mul(const Tensor<T>& a, const Tensor<T>& b) {
    Shape result_shape = detail::broadcast_shapes(a.shape(), b.shape());
    Tensor<T> result(result_shape);

    detail::iterate_indices(result_shape, [&](const std::vector<size_t>& indices) {
        auto a_indices = detail::get_broadcast_indices(indices, a.shape(), result_shape);
        auto b_indices = detail::get_broadcast_indices(indices, b.shape(), result_shape);
        result[indices] = a[a_indices] * b[b_indices];
    });

    return result;
}

/**
 * @brief Element-wise division with broadcasting support
 *
 * @tparam T Element type
 * @param a First tensor (dividend)
 * @param b Second tensor (divisor)
 * @return Result tensor with broadcasted shape
 * @throws std::invalid_argument if shapes are not broadcastable
 */
template <typename T>
Tensor<T> div(const Tensor<T>& a, const Tensor<T>& b) {
    Shape result_shape = detail::broadcast_shapes(a.shape(), b.shape());
    Tensor<T> result(result_shape);

    detail::iterate_indices(result_shape, [&](const std::vector<size_t>& indices) {
        auto a_indices = detail::get_broadcast_indices(indices, a.shape(), result_shape);
        auto b_indices = detail::get_broadcast_indices(indices, b.shape(), result_shape);
        result[indices] = a[a_indices] / b[b_indices];
    });

    return result;
}

/**
 * @brief Element-wise exponential function
 *
 * @tparam T Element type
 * @param a Input tensor
 * @return Result tensor with exp applied element-wise
 */
template <typename T>
Tensor<T> exp(const Tensor<T>& a) {
    Tensor<T> result(a.shape());

    detail::iterate_indices(a.shape(), [&](const std::vector<size_t>& indices) {
        result[indices] = std::exp(a[indices]);
    });

    return result;
}

/**
 * @brief Element-wise natural logarithm
 *
 * @tparam T Element type
 * @param a Input tensor
 * @return Result tensor with log applied element-wise
 */
template <typename T>
Tensor<T> log(const Tensor<T>& a) {
    Tensor<T> result(a.shape());

    detail::iterate_indices(a.shape(), [&](const std::vector<size_t>& indices) {
        result[indices] = std::log(a[indices]);
    });

    return result;
}

/**
 * @brief Element-wise square root
 *
 * @tparam T Element type
 * @param a Input tensor
 * @return Result tensor with sqrt applied element-wise
 */
template <typename T>
Tensor<T> sqrt(const Tensor<T>& a) {
    Tensor<T> result(a.shape());

    detail::iterate_indices(a.shape(), [&](const std::vector<size_t>& indices) {
        result[indices] = std::sqrt(a[indices]);
    });

    return result;
}

/**
 * @brief Element-wise power function
 *
 * @tparam T Element type
 * @param a Base tensor
 * @param b Exponent tensor (supports broadcasting)
 * @return Result tensor with pow applied element-wise
 * @throws std::invalid_argument if shapes are not broadcastable
 */
template <typename T>
Tensor<T> pow(const Tensor<T>& a, const Tensor<T>& b) {
    Shape result_shape = detail::broadcast_shapes(a.shape(), b.shape());
    Tensor<T> result(result_shape);

    detail::iterate_indices(result_shape, [&](const std::vector<size_t>& indices) {
        auto a_indices = detail::get_broadcast_indices(indices, a.shape(), result_shape);
        auto b_indices = detail::get_broadcast_indices(indices, b.shape(), result_shape);
        result[indices] = std::pow(a[a_indices], b[b_indices]);
    });

    return result;
}

/**
 * @brief Element-wise power function with scalar exponent
 *
 * @tparam T Element type
 * @param a Base tensor
 * @param exponent Scalar exponent
 * @return Result tensor with pow applied element-wise
 */
template <typename T>
Tensor<T> pow(const Tensor<T>& a, T exponent) {
    Tensor<T> result(a.shape());

    detail::iterate_indices(a.shape(), [&](const std::vector<size_t>& indices) {
        result[indices] = std::pow(a[indices], exponent);
    });

    return result;
}

}  // namespace gradflow
