#pragma once

#include <cstring>
#include <memory>
#include <stdexcept>

#include "allocator.hpp"
#include "device.hpp"

namespace gradflow {

/**
 * @brief Template class for managing memory buffers
 *
 * Storage<T> manages the actual memory buffer for tensor data.
 * It uses RAII to ensure proper resource management and supports
 * different device allocators.
 *
 * @tparam T Element type
 */
template <typename T>
class Storage {
public:
    /**
     * @brief Constructs an empty storage
     */
    Storage() : data_(nullptr), size_(0), allocator_(nullptr) {}

    /**
     * @brief Constructs a storage with the specified size and allocator
     * @param size Number of elements
     * @param allocator Device allocator (defaults to CPU allocator)
     */
    explicit Storage(size_t size, std::shared_ptr<DeviceAllocator> allocator = nullptr)
        : size_(size), allocator_(allocator ? allocator : get_default_cpu_allocator()) {
        if (size > 0) {
            data_ = static_cast<T*>(allocator_->allocate(size * sizeof(T)));
        } else {
            data_ = nullptr;
        }
    }

    /**
     * @brief Destructor - deallocates memory
     */
    ~Storage() {
        if (data_ != nullptr && allocator_ != nullptr) {
            allocator_->deallocate(data_);
        }
    }

    // Disable copy constructor and assignment
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    /**
     * @brief Move constructor
     */
    Storage(Storage&& other) noexcept
        : data_(other.data_), size_(other.size_), allocator_(std::move(other.allocator_)) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    /**
     * @brief Move assignment operator
     */
    Storage& operator=(Storage&& other) noexcept {
        if (this != &other) {
            // Clean up existing resources
            if (data_ != nullptr && allocator_ != nullptr) {
                allocator_->deallocate(data_);
            }

            // Transfer ownership
            data_ = other.data_;
            size_ = other.size_;
            allocator_ = std::move(other.allocator_);

            // Reset other
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    /**
     * @brief Returns a pointer to the data
     * @return Data pointer
     */
    T* data() { return data_; }

    /**
     * @brief Returns a const pointer to the data
     * @return Const data pointer
     */
    const T* data() const { return data_; }

    /**
     * @brief Returns the number of elements
     * @return Number of elements
     */
    size_t size() const { return size_; }

    /**
     * @brief Checks if the storage is empty
     * @return True if empty
     */
    bool empty() const { return size_ == 0 || data_ == nullptr; }

    /**
     * @brief Access element at index (no bounds checking)
     * @param index Element index
     * @return Reference to element
     */
    T& operator[](size_t index) { return data_[index]; }

    /**
     * @brief Access element at index (no bounds checking, const version)
     * @param index Element index
     * @return Const reference to element
     */
    const T& operator[](size_t index) const { return data_[index]; }

    /**
     * @brief Access element at index (with bounds checking)
     * @param index Element index
     * @return Reference to element
     * @throws std::out_of_range if index is out of bounds
     */
    T& at(size_t index) {
        if (index >= size_) {
            throw std::out_of_range("Storage index out of range");
        }
        return data_[index];
    }

    /**
     * @brief Access element at index (with bounds checking, const version)
     * @param index Element index
     * @return Const reference to element
     * @throws std::out_of_range if index is out of bounds
     */
    const T& at(size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("Storage index out of range");
        }
        return data_[index];
    }

    /**
     * @brief Fills the storage with a value
     * @param value Value to fill with
     */
    void fill(const T& value) {
        for (size_t i = 0; i < size_; ++i) {
            data_[i] = value;
        }
    }

    /**
     * @brief Returns the device this storage is on
     * @return Device
     */
    Device device() const {
        if (allocator_ == nullptr) {
            return cpu();
        }
        return allocator_->device();
    }

    /**
     * @brief Returns the allocator used by this storage
     * @return Shared pointer to allocator
     */
    std::shared_ptr<DeviceAllocator> allocator() const { return allocator_; }

    /**
     * @brief Copies data from another storage
     *
     * Both storages must have the same size and be on the same device.
     *
     * @param other Source storage
     * @throws std::invalid_argument if sizes don't match
     */
    void copy_from(const Storage<T>& other) {
        if (size_ != other.size_) {
            throw std::invalid_argument("Storage sizes must match for copy");
        }
        if (size_ == 0) {
            return;
        }
        std::memcpy(data_, other.data_, size_ * sizeof(T));
    }

private:
    T* data_;
    size_t size_;
    std::shared_ptr<DeviceAllocator> allocator_;
};

}  // namespace gradflow
