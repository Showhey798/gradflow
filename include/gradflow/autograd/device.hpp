#pragma once

#include <memory>
#include <string>

namespace gradflow {

/**
 * @brief Enum representing different device types
 */
enum class DeviceType { CPU, CUDA, METAL };

/**
 * @brief Represents a device (CPU, GPU, etc.)
 *
 * Device encapsulates information about where tensor data is stored.
 */
class Device {
public:
    /**
     * @brief Constructs a device with the specified type
     * @param type The device type
     * @param index The device index (default: 0)
     */
    explicit Device(DeviceType type, int index = 0) : type_(type), index_(index) {}

    /**
     * @brief Returns the device type
     * @return Device type
     */
    [[nodiscard]] DeviceType type() const { return type_; }

    /**
     * @brief Returns the device index
     * @return Device index
     */
    [[nodiscard]] int index() const { return index_; }

    /**
     * @brief Checks if this device is a CPU
     * @return True if device is CPU
     */
    [[nodiscard]] bool isCpu() const { return type_ == DeviceType::CPU; }

    /**
     * @brief Checks if this device is a CUDA device
     * @return True if device is CUDA
     */
    [[nodiscard]] bool isCuda() const { return type_ == DeviceType::CUDA; }

    /**
     * @brief Checks if this device is a Metal device
     * @return True if device is Metal
     */
    [[nodiscard]] bool isMetal() const { return type_ == DeviceType::METAL; }

    /**
     * @brief Equality comparison
     */
    bool operator==(const Device& other) const {
        return type_ == other.type_ && index_ == other.index_;
    }

    /**
     * @brief Inequality comparison
     */
    bool operator!=(const Device& other) const { return !(*this == other); }

    /**
     * @brief Returns a string representation of the device
     * @return String representation
     */
    [[nodiscard]] std::string toString() const {
        std::string type_str;
        switch (type_) {
            case DeviceType::CPU:
                type_str = "cpu";
                break;
            case DeviceType::CUDA:
                type_str = "cuda";
                break;
            case DeviceType::METAL:
                type_str = "metal";
                break;
        }
        return type_str + ":" + std::to_string(index_);
    }

private:
    DeviceType type_;
    int index_;
};

/**
 * @brief Factory function to create a CPU device
 * @return CPU device
 */
inline Device cpu() {
    return Device(DeviceType::CPU, 0);
}

/**
 * @brief Factory function to create a CUDA device
 * @param index Device index (default: 0)
 * @return CUDA device
 */
inline Device cuda(int index = 0) {
    return Device(DeviceType::CUDA, index);
}

/**
 * @brief Factory function to create a Metal device
 * @param index Device index (default: 0)
 * @return Metal device
 */
inline Device metal(int index = 0) {
    return Device(DeviceType::METAL, index);
}

// Forward declaration
class DeviceAllocator;

/**
 * @brief Manages devices and their allocators
 *
 * DeviceManager provides centralized device management including:
 * - Device availability checking
 * - Device enumeration
 * - Allocator management and caching
 */
class DeviceManager {
public:
    /**
     * @brief Returns the default device (CPU:0)
     * @return Default CPU device
     */
    static Device getDefaultDevice();

    /**
     * @brief Returns a device of the specified type and index
     * @param type Device type
     * @param index Device index (default: 0)
     * @return Device instance
     */
    static Device getDevice(DeviceType type, int index = 0);

    /**
     * @brief Checks if a device is available
     * @param type Device type
     * @param index Device index
     * @return True if device is available
     */
    static bool isDeviceAvailable(DeviceType type, int index);

    /**
     * @brief Returns the number of devices of the specified type
     * @param type Device type
     * @return Number of devices
     */
    static int getDeviceCount(DeviceType type);

    /**
     * @brief Returns an allocator for the specified device
     *
     * Allocators are cached per device. Multiple calls for the same
     * device will return the same allocator instance.
     *
     * @param device Device to get allocator for
     * @return Shared pointer to device allocator
     */
    static std::shared_ptr<DeviceAllocator> getAllocator(const Device& device);

private:
    DeviceManager() = delete;  // Static class, no instantiation
};

}  // namespace gradflow
