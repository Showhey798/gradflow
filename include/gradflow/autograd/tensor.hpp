#pragma once

#include "allocator.hpp"
#include "device.hpp"
#include "shape.hpp"
#include "storage.hpp"

#include <initializer_list>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace gradflow {

/**
 * @brief Template class representing a multidimensional array (tensor)
 *
 * Tensor<T> is the fundamental data structure for representing N-dimensional arrays.
 * It supports various operations like reshaping, transposing, slicing, and more.
 *
 * Design:
 * - Storage: Shared pointer to memory buffer (reference counting)
 * - Shape: Dimensions of the tensor
 * - Stride: Memory layout (enables zero-copy view operations)
 * - Offset: Starting position in storage (for slicing)
 *
 * @tparam T Element type
 */
template <typename T>
class Tensor {
public:
    /**
     * @brief Default constructor - creates an empty tensor
     */
    Tensor() : storage_(nullptr), stride_(Shape()), offset_(0) {}

    /**
     * @brief Constructs a tensor with the specified shape
     *
     * Memory is allocated but not initialized.
     *
     * @param shape Shape of the tensor
     * @param allocator Device allocator (defaults to CPU allocator)
     */
    explicit Tensor(const Shape& shape, const std::shared_ptr<DeviceAllocator>& allocator = nullptr)
        : storage_(std::make_shared<Storage<T>>(shape.size(), allocator)),
          shape_(shape),
          stride_(shape),
          offset_(0) {}

    /**
     * @brief Constructs a tensor from an initializer list (1D)
     *
     * @param data Initializer list of data
     * @param allocator Device allocator (defaults to CPU allocator)
     */
    Tensor(std::initializer_list<T> data,
           const std::shared_ptr<DeviceAllocator>& allocator = nullptr)
        : storage_(std::make_shared<Storage<T>>(data.size(), allocator)),
          shape_({data.size()}),
          stride_(shape_),
          offset_(0) {
        size_t i = 0;
        for (const auto& value : data) {
            (*storage_)[i++] = value;
        }
    }

    /**
     * @brief Constructs a tensor from shape and data vector
     *
     * @param shape Shape of the tensor
     * @param data Vector of data (must match shape.size())
     * @param allocator Device allocator (defaults to CPU allocator)
     * @throws std::invalid_argument if data size doesn't match shape
     */
    Tensor(const Shape& shape,
           const std::vector<T>& data,
           const std::shared_ptr<DeviceAllocator>& allocator = nullptr)
        : storage_(std::make_shared<Storage<T>>(shape.size(), allocator)),
          shape_(shape),
          stride_(shape),
          offset_(0) {
        if (data.size() != shape.size()) {
            throw std::invalid_argument("Data size must match shape size");
        }
        for (size_t i = 0; i < data.size(); ++i) {
            (*storage_)[i] = data[i];
        }
    }

    /**
     * @brief Constructs a tensor from nested initializer lists (2D)
     *
     * @param data Nested initializer list
     * @param allocator Device allocator (defaults to CPU allocator)
     */
    Tensor(std::initializer_list<std::initializer_list<T>> data,
           const std::shared_ptr<DeviceAllocator>& allocator = nullptr)
        : shape_({data.size(), data.begin()->size()}),
          stride_(shape_),
          offset_(0) {
        size_t rows = data.size();
        if (rows == 0) {
            throw std::invalid_argument("Cannot create tensor from empty data");
        }

        size_t cols = data.begin()->size();
        for (const auto& row : data) {
            if (row.size() != cols) {
                throw std::invalid_argument("All rows must have the same size");
            }
        }

        storage_ = std::make_shared<Storage<T>>(shape_.size(), allocator);

        size_t idx = 0;
        for (const auto& row : data) {
            for (const auto& value : row) {
                (*storage_)[idx++] = value;
            }
        }
    }

    // Copy constructor
    Tensor(const Tensor& other) = default;

    // Copy assignment
    Tensor& operator=(const Tensor& other) = default;

    // Move constructor
    Tensor(Tensor&& other) noexcept = default;

    // Move assignment
    Tensor& operator=(Tensor&& other) noexcept = default;

    // Destructor
    ~Tensor() = default;

    /**
     * @brief Returns the shape of the tensor
     * @return Shape reference
     */
    [[nodiscard]] const Shape& shape() const { return shape_; }

    /**
     * @brief Returns the stride of the tensor
     * @return Stride reference
     */
    [[nodiscard]] const Stride& stride() const { return stride_; }

    /**
     * @brief Returns the number of dimensions
     * @return Number of dimensions
     */
    [[nodiscard]] size_t ndim() const { return shape_.ndim(); }

    /**
     * @brief Returns the total number of elements
     * @return Total number of elements
     */
    [[nodiscard]] size_t size() const { return shape_.size(); }

    /**
     * @brief Returns the device this tensor is on
     * @return Device
     */
    [[nodiscard]] Device device() const {
        if (storage_ == nullptr) {
            return cpu();
        }
        return storage_->device();
    }

    /**
     * @brief Access element at multidimensional index
     *
     * @param indices Vector of indices
     * @return Reference to element
     * @throws std::out_of_range if indices are out of bounds
     */
    T& operator[](const std::vector<size_t>& indices) {
        if (indices.size() != ndim()) {
            throw std::out_of_range("Number of indices must match number of dimensions");
        }
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds");
            }
        }
        const size_t kLinearIndex = stride_.offset(indices) + offset_;
        return (*storage_)[kLinearIndex];
    }

    /**
     * @brief Access element at multidimensional index (const version)
     */
    const T& operator[](const std::vector<size_t>& indices) const {
        if (indices.size() != ndim()) {
            throw std::out_of_range("Number of indices must match number of dimensions");
        }
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds");
            }
        }
        const size_t kLinearIndex = stride_.offset(indices) + offset_;
        return (*storage_)[kLinearIndex];
    }

    /**
     * @brief Returns a pointer to the underlying data
     * @return Data pointer (nullptr if tensor is empty)
     */
    T* data() {
        if (storage_ == nullptr) {
            return nullptr;
        }
        return storage_->data() + offset_;
    }

    /**
     * @brief Returns a const pointer to the underlying data
     */
    const T* data() const {
        if (storage_ == nullptr) {
            return nullptr;
        }
        return storage_->data() + offset_;
    }

    /**
     * @brief Checks if the tensor is contiguous in memory
     *
     * A tensor is contiguous if its stride matches the row-major layout.
     *
     * @return True if contiguous
     */
    [[nodiscard]] bool isContiguous() const { return stride_.isContiguous(shape_); }

    /**
     * @brief Returns a contiguous copy of the tensor
     *
     * If the tensor is already contiguous, returns a copy.
     * Otherwise, creates a new contiguous tensor with the same data.
     *
     * @return Contiguous tensor
     */
    Tensor<T> contiguous() const {
        if (isContiguous() && offset_ == 0) {
            return *this;
        }

        // Create a new contiguous tensor
        Tensor<T> result(shape_, storage_->allocator());

        // Copy elements in the correct order
        copyToContiguous(result);

        return result;
    }

    /**
     * @brief Returns a view of the tensor with a new shape (zero-copy)
     *
     * The total number of elements must remain the same.
     * The tensor must be contiguous.
     *
     * @param new_shape New shape
     * @return Tensor view with new shape
     * @throws std::invalid_argument if tensor is not contiguous or sizes don't match
     */
    Tensor<T> view(const Shape& new_shape) const {
        if (!isContiguous()) {
            throw std::invalid_argument("Tensor must be contiguous for view operation");
        }
        if (new_shape.size() != size()) {
            throw std::invalid_argument("Total number of elements must remain the same");
        }

        Tensor<T> result;
        result.storage_ = storage_;
        result.shape_ = new_shape;
        result.stride_ = Stride(new_shape);
        result.offset_ = offset_;

        return result;
    }

    /**
     * @brief Reshapes the tensor (may require copy if not contiguous)
     *
     * @param new_shape New shape
     * @return Reshaped tensor
     */
    Tensor<T> reshape(const Shape& new_shape) const {
        if (new_shape.size() != size()) {
            throw std::invalid_argument("Total number of elements must remain the same");
        }

        if (isContiguous()) {
            return view(new_shape);
        } else {
            // Need to make contiguous first
            return contiguous().view(new_shape);
        }
    }

    /**
     * @brief Transposes two dimensions (zero-copy)
     *
     * @param dim0 First dimension
     * @param dim1 Second dimension
     * @return Transposed tensor view
     * @throws std::out_of_range if dimensions are out of bounds
     */
    Tensor<T> transpose(size_t dim0, size_t dim1) const {
        if (dim0 >= ndim() || dim1 >= ndim()) {
            throw std::out_of_range("Dimension out of range");
        }

        if (dim0 == dim1) {
            return *this;
        }

        Tensor<T> result;
        result.storage_ = storage_;
        result.shape_ = shape_;
        result.stride_ = stride_;
        result.offset_ = offset_;

        // Swap dimensions in shape
        auto new_dims = result.shape_.dims();
        std::swap(new_dims[dim0], new_dims[dim1]);
        result.shape_ = Shape(new_dims);

        // Swap strides
        auto new_strides = result.stride_.strides();
        std::swap(new_strides[dim0], new_strides[dim1]);
        result.stride_ = Stride(new_strides);

        return result;
    }

    /**
     * @brief Permutes dimensions according to the given order (zero-copy)
     *
     * @param dims New order of dimensions
     * @return Permuted tensor view
     * @throws std::invalid_argument if dims is invalid
     */
    Tensor<T> permute(const std::vector<size_t>& dims) const {
        if (dims.size() != ndim()) {
            throw std::invalid_argument("Number of dimensions must match");
        }

        // Check that dims is a valid permutation
        std::vector<bool> seen(ndim(), false);
        for (const size_t kDim : dims) {
            if (kDim >= ndim() || seen[kDim]) {
                throw std::invalid_argument("Invalid permutation");
            }
            seen[kDim] = true;
        }

        Tensor<T> result;
        result.storage_ = storage_;
        result.offset_ = offset_;

        // Reorder shape and stride
        std::vector<size_t> new_dims(ndim());
        std::vector<size_t> new_strides(ndim());
        for (size_t i = 0; i < ndim(); ++i) {
            new_dims[i] = shape_[dims[i]];
            new_strides[i] = stride_[dims[i]];
        }

        result.shape_ = Shape(new_dims);
        result.stride_ = Stride(new_strides);

        return result;
    }

    /**
     * @brief Slices the tensor along a dimension (zero-copy)
     *
     * @param dim Dimension to slice
     * @param start Start index (inclusive)
     * @param end End index (exclusive)
     * @return Sliced tensor view
     * @throws std::out_of_range if parameters are out of bounds
     */
    Tensor<T> slice(size_t dim, size_t start, size_t end) const {
        if (dim >= ndim()) {
            throw std::out_of_range("Dimension out of range");
        }
        if (start >= shape_[dim] || end > shape_[dim] || start >= end) {
            throw std::out_of_range("Invalid slice range");
        }

        Tensor<T> result;
        result.storage_ = storage_;
        result.stride_ = stride_;

        // Calculate new offset
        std::vector<size_t> start_indices(ndim(), 0);
        start_indices[dim] = start;
        result.offset_ = offset_ + stride_.offset(start_indices);

        // Update shape
        auto new_dims = shape_.dims();
        new_dims[dim] = end - start;
        result.shape_ = Shape(new_dims);

        return result;
    }

    // Factory functions

    /**
     * @brief Creates a tensor filled with zeros
     *
     * @param shape Shape of the tensor
     * @param allocator Device allocator (defaults to CPU allocator)
     * @return Tensor filled with zeros
     */
    static Tensor<T> zeros(const Shape& shape,
                           const std::shared_ptr<DeviceAllocator>& allocator = nullptr) {
        Tensor<T> result(shape, allocator);
        result.storage_->fill(T(0));
        return result;
    }

    /**
     * @brief Creates a tensor filled with ones
     *
     * @param shape Shape of the tensor
     * @param allocator Device allocator (defaults to CPU allocator)
     * @return Tensor filled with ones
     */
    static Tensor<T> ones(const Shape& shape,
                          const std::shared_ptr<DeviceAllocator>& allocator = nullptr) {
        Tensor<T> result(shape, allocator);
        result.storage_->fill(T(1));
        return result;
    }

    /**
     * @brief Sets the random seed for reproducible random number generation
     *
     * This affects all subsequent calls to randn() and rand().
     * Should be called at the beginning of tests or experiments for reproducibility.
     *
     * @param seed Random seed value
     */
    static void setSeed(unsigned int seed) { getRandomGenerator().seed(seed); }

    /**
     * @brief Creates a tensor filled with values from a normal distribution
     *
     * @param shape Shape of the tensor
     * @param mean Mean of the distribution
     * @param stddev Standard deviation of the distribution
     * @param allocator Device allocator (defaults to CPU allocator)
     * @return Tensor with random values
     */
    static Tensor<T> randn(const Shape& shape,
                           T mean = T(0),
                           T stddev = T(1),
                           const std::shared_ptr<DeviceAllocator>& allocator = nullptr) {
        Tensor<T> result(shape, allocator);

        std::normal_distribution<T> dist(mean, stddev);

        for (size_t i = 0; i < result.size(); ++i) {
            (*result.storage_)[i] = dist(getRandomGenerator());
        }

        return result;
    }

    /**
     * @brief Creates a tensor filled with random values from uniform distribution [0, 1)
     *
     * @param shape Shape of the tensor
     * @param allocator Device allocator (defaults to CPU allocator)
     * @return Tensor with random values
     */
    static Tensor<T> rand(const Shape& shape,
                          const std::shared_ptr<DeviceAllocator>& allocator = nullptr) {
        Tensor<T> result(shape, allocator);

        std::uniform_real_distribution<T> dist(T(0), T(1));

        for (size_t i = 0; i < result.size(); ++i) {
            (*result.storage_)[i] = dist(getRandomGenerator());
        }

        return result;
    }

    /**
     * @brief Creates an identity matrix
     *
     * @param n Size of the matrix
     * @param allocator Device allocator (defaults to CPU allocator)
     * @return Identity matrix of shape [n, n]
     */
    static Tensor<T> eye(size_t n, const std::shared_ptr<DeviceAllocator>& allocator = nullptr) {
        Tensor<T> result = zeros(Shape({n, n}), allocator);

        for (size_t i = 0; i < n; ++i) {
            result[{i, i}] = T(1);
        }

        return result;
    }

    /**
     * @brief Creates a tensor like another tensor (same shape and device)
     *
     * @param other Template tensor
     * @return New tensor with same shape and device
     */
    static Tensor<T> zerosLike(const Tensor<T>& other) {
        return zeros(other.shape(), other.storage_->allocator());
    }

    /**
     * @brief Creates a tensor like another tensor filled with ones
     *
     * @param other Template tensor
     * @return New tensor with same shape and device
     */
    static Tensor<T> onesLike(const Tensor<T>& other) {
        return ones(other.shape(), other.storage_->allocator());
    }

private:
    std::shared_ptr<Storage<T>> storage_;
    Shape shape_;
    Stride stride_;
    size_t offset_;

    /**
     * @brief Internal constructor for creating views
     */
    Tensor(std::shared_ptr<Storage<T>> storage, Shape shape, Stride stride, size_t offset)
        : storage_(std::move(storage)),
          shape_(std::move(shape)),
          stride_(std::move(stride)),
          offset_(offset) {}

    /**
     * @brief Helper function to copy data to a contiguous tensor
     */
    void copyToContiguous(Tensor<T>& dest) const {
        // Recursive helper to iterate through all indices
        std::vector<size_t> indices(ndim(), 0);
        copyRecursive(dest, indices, 0);
    }

    void copyRecursive(Tensor<T>& dest, std::vector<size_t>& indices, size_t dim) const {
        if (dim == ndim()) {
            // Base case: copy the element
            dest[indices] = (*this)[indices];
            return;
        }

        // Recursive case: iterate through current dimension
        for (size_t i = 0; i < shape_[dim]; ++i) {
            indices[dim] = i;
            copyRecursive(dest, indices, dim + 1);
        }
    }

    /**
     * @brief Returns the static random number generator
     *
     * This generator is shared across all Tensor instances and can be seeded
     * using setSeed() for reproducible random number generation.
     *
     * @return Reference to the static random generator
     */
    static std::mt19937& getRandomGenerator() {
        static std::mt19937 generator(std::random_device{}());
        return generator;
    }
};

}  // namespace gradflow
