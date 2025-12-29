#pragma once

#include <string>

namespace gradflow {

/**
 * @brief Enum representing different device types
 */
enum class DeviceType { CPU, CUDA, Metal };

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
    DeviceType type() const { return type_; }

    /**
     * @brief Returns the device index
     * @return Device index
     */
    int index() const { return index_; }

    /**
     * @brief Checks if this device is a CPU
     * @return True if device is CPU
     */
    bool is_cpu() const { return type_ == DeviceType::CPU; }

    /**
     * @brief Checks if this device is a CUDA device
     * @return True if device is CUDA
     */
    bool is_cuda() const { return type_ == DeviceType::CUDA; }

    /**
     * @brief Checks if this device is a Metal device
     * @return True if device is Metal
     */
    bool is_metal() const { return type_ == DeviceType::Metal; }

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
    std::string to_string() const {
        std::string type_str;
        switch (type_) {
            case DeviceType::CPU:
                type_str = "cpu";
                break;
            case DeviceType::CUDA:
                type_str = "cuda";
                break;
            case DeviceType::Metal:
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
    return Device(DeviceType::Metal, index);
}

}  // namespace gradflow
