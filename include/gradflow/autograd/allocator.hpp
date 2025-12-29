#pragma once

#include "device.hpp"

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <stdexcept>

namespace gradflow {

/**
 * @brief Abstract base class for device-specific memory allocation
 *
 * DeviceAllocator provides an interface for allocating and deallocating
 * memory on different devices (CPU, GPU, etc.).
 */
class DeviceAllocator {
public:
    virtual ~DeviceAllocator() = default;

    /**
     * @brief Allocates memory on the device
     * @param size Number of bytes to allocate
     * @return Pointer to allocated memory
     * @throws std::bad_alloc if allocation fails
     */
    virtual void* allocate(size_t size) = 0;

    /**
     * @brief Deallocates memory on the device
     * @param ptr Pointer to memory to deallocate
     */
    virtual void deallocate(void* ptr) = 0;

    /**
     * @brief Returns the device this allocator manages
     * @return Device reference
     */
    virtual Device device() const = 0;

    /**
     * @brief Returns the alignment requirement for allocations
     * @return Alignment in bytes
     */
    virtual size_t alignment() const = 0;
};

/**
 * @brief CPU memory allocator
 *
 * Allocates aligned memory on the CPU using aligned_alloc.
 */
class CPUAllocator : public DeviceAllocator {
public:
    /**
     * @brief Default alignment for CPU allocations (64 bytes for SIMD)
     */
    static constexpr size_t DEFAULT_ALIGNMENT = 64;

    /**
     * @brief Constructs a CPU allocator with optional custom alignment
     * @param alignment Alignment requirement (must be a power of 2)
     */
    explicit CPUAllocator(size_t alignment = DEFAULT_ALIGNMENT) : alignment_(alignment) {
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            throw std::invalid_argument("Alignment must be a power of 2");
        }
    }

    /**
     * @brief Allocates aligned memory on the CPU
     * @param size Number of bytes to allocate
     * @return Pointer to allocated memory
     * @throws std::bad_alloc if allocation fails
     */
    void* allocate(size_t size) override {
        if (size == 0) {
            return nullptr;
        }

        // Round up size to be a multiple of alignment
        size_t aligned_size = (size + alignment_ - 1) & ~(alignment_ - 1);

#if defined(_WIN32) || defined(_WIN64)
        void* ptr = _aligned_malloc(aligned_size, alignment_);
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }
#else
        void* ptr = nullptr;
        int result = posix_memalign(&ptr, alignment_, aligned_size);
        if (result != 0) {
            throw std::bad_alloc();
        }
#endif
        return ptr;
    }

    /**
     * @brief Deallocates memory allocated by this allocator
     * @param ptr Pointer to memory to deallocate
     */
    void deallocate(void* ptr) override {
        if (ptr == nullptr) {
            return;
        }

#if defined(_WIN32) || defined(_WIN64)
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }

    /**
     * @brief Returns the CPU device
     * @return CPU device
     */
    Device device() const override { return cpu(); }

    /**
     * @brief Returns the alignment requirement
     * @return Alignment in bytes
     */
    size_t alignment() const override { return alignment_; }

private:
    size_t alignment_;
};

/**
 * @brief Returns a shared pointer to the default CPU allocator
 * @return Shared pointer to CPU allocator
 */
inline std::shared_ptr<DeviceAllocator> get_default_cpu_allocator() {
    static auto allocator = std::make_shared<CPUAllocator>();
    return allocator;
}

}  // namespace gradflow
