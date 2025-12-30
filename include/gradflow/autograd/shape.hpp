#pragma once

#include <algorithm>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace gradflow {

/**
 * @brief Represents the shape of a multidimensional array
 *
 * Shape stores the dimensions of a tensor and provides utilities
 * for broadcasting and size calculations.
 */
class Shape {
public:
    /**
     * @brief Default constructor - creates a scalar shape (0-dimensional)
     */
    Shape() = default;

    /**
     * @brief Constructs a shape from an initializer list
     * @param dims List of dimensions
     */
    Shape(std::initializer_list<size_t> dims) : dims_(dims) {}

    /**
     * @brief Constructs a shape from a vector
     * @param dims Vector of dimensions
     */
    explicit Shape(std::vector<size_t> dims) : dims_(std::move(dims)) {}

    /**
     * @brief Returns the number of dimensions
     * @return Number of dimensions
     */
    [[nodiscard]] size_t ndim() const { return dims_.size(); }

    /**
     * @brief Returns the total number of elements
     * @return Total number of elements (product of all dimensions)
     */
    [[nodiscard]] size_t size() const {
        if (dims_.empty()) {
            return 1;  // Scalar has size 1
        }
        return std::accumulate(
            dims_.begin(), dims_.end(), static_cast<size_t>(1), std::multiplies<>());
    }

    /**
     * @brief Access dimension at index (no bounds checking)
     * @param index Dimension index
     * @return Dimension value
     */
    [[nodiscard]] size_t operator[](size_t index) const { return dims_[index]; }

    /**
     * @brief Access dimension at index (with bounds checking)
     * @param index Dimension index
     * @return Dimension value
     * @throws std::out_of_range if index is out of bounds
     */
    [[nodiscard]] size_t at(size_t index) const {
        if (index >= dims_.size()) {
            throw std::out_of_range("Shape index out of range");
        }
        return dims_[index];
    }

    /**
     * @brief Equality comparison
     */
    bool operator==(const Shape& other) const { return dims_ == other.dims_; }

    /**
     * @brief Inequality comparison
     */
    bool operator!=(const Shape& other) const { return dims_ != other.dims_; }

    /**
     * @brief Checks if this shape is broadcastable with another shape
     *
     * Broadcasting rules (from right to left):
     * - Dimensions must be equal, or
     * - One of them is 1, or
     * - One of them is missing (different ndim)
     *
     * @param other The other shape to check against
     * @return True if shapes are broadcastable
     */
    [[nodiscard]] bool isBroadcastableWith(const Shape& other) const {
        const size_t kNdim1 = ndim();
        const size_t kNdim2 = other.ndim();
        const size_t kMaxNdim = std::max(kNdim1, kNdim2);

        for (size_t i = 0; i < kMaxNdim; ++i) {
            const size_t kDim1 = i < kNdim1 ? dims_[kNdim1 - 1 - i] : 1;
            const size_t kDim2 = i < kNdim2 ? other.dims_[kNdim2 - 1 - i] : 1;

            if (kDim1 != kDim2 && kDim1 != 1 && kDim2 != 1) {
                return false;
            }
        }

        return true;
    }

    /**
     * @brief Computes the broadcast shape with another shape
     *
     * @param other The other shape to broadcast with
     * @return The resulting broadcast shape
     * @throws std::invalid_argument if shapes are not broadcastable
     */
    [[nodiscard]] Shape broadcastWith(const Shape& other) const {
        if (!isBroadcastableWith(other)) {
            throw std::invalid_argument("Shapes are not broadcastable");
        }

        const size_t kNdim1 = ndim();
        const size_t kNdim2 = other.ndim();
        const size_t kMaxNdim = std::max(kNdim1, kNdim2);

        std::vector<size_t> result_dims(kMaxNdim);

        for (size_t i = 0; i < kMaxNdim; ++i) {
            const size_t kDim1 = i < kNdim1 ? dims_[kNdim1 - 1 - i] : 1;
            const size_t kDim2 = i < kNdim2 ? other.dims_[kNdim2 - 1 - i] : 1;

            result_dims[kMaxNdim - 1 - i] = std::max(kDim1, kDim2);
        }

        return Shape(result_dims);
    }

    /**
     * @brief Returns the underlying dimensions vector
     * @return Vector of dimensions
     */
    [[nodiscard]] const std::vector<size_t>& dims() const { return dims_; }

private:
    std::vector<size_t> dims_;
};

/**
 * @brief Represents the stride (memory layout) of a multidimensional array
 *
 * Stride defines how to navigate through memory to access elements.
 * For row-major layout: stride[i] = product of all dimensions after i.
 */
class Stride {
public:
    /**
     * @brief Constructs row-major strides from a shape
     * @param shape The shape to compute strides for
     */
    explicit Stride(const Shape& shape) {
        const size_t kNdim = shape.ndim();
        if (kNdim == 0) {
            return;  // Scalar has no strides
        }

        strides_.resize(kNdim);
        strides_[kNdim - 1] = 1;

        for (size_t i = kNdim - 1; i > 0; --i) {
            strides_[i - 1] = strides_[i] * shape[i];
        }
    }

    /**
     * @brief Constructs strides from a vector of stride values
     * @param strides Vector of stride values
     */
    explicit Stride(std::vector<size_t> strides) : strides_(std::move(strides)) {}

    /**
     * @brief Returns the number of dimensions
     * @return Number of dimensions
     */
    [[nodiscard]] size_t ndim() const { return strides_.size(); }

    /**
     * @brief Access stride at index (no bounds checking)
     * @param index Dimension index
     * @return Stride value
     */
    [[nodiscard]] size_t operator[](size_t index) const { return strides_[index]; }

    /**
     * @brief Computes the linear offset for a given multidimensional index
     *
     * @param indices Multidimensional indices
     * @return Linear offset in memory
     */
    [[nodiscard]] size_t offset(const std::vector<size_t>& indices) const {
        size_t result = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            result += indices[i] * strides_[i];
        }
        return result;
    }

    /**
     * @brief Checks if the stride represents a contiguous memory layout
     *
     * A stride is contiguous if it matches the row-major layout for the given shape.
     *
     * @param shape The shape to check against
     * @return True if contiguous
     */
    [[nodiscard]] bool isContiguous(const Shape& shape) const {
        if (ndim() != shape.ndim()) {
            return false;
        }

        if (ndim() == 0) {
            return true;  // Scalar is always contiguous
        }

        // Check if this stride matches the row-major stride
        Stride row_major(shape);
        for (size_t i = 0; i < ndim(); ++i) {
            if (strides_[i] != row_major.strides_[i]) {
                return false;
            }
        }

        return true;
    }

    /**
     * @brief Returns the underlying strides vector
     * @return Vector of stride values
     */
    [[nodiscard]] const std::vector<size_t>& strides() const { return strides_; }

private:
    std::vector<size_t> strides_;
};

}  // namespace gradflow
