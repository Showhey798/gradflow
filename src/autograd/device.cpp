#include "gradflow/autograd/device.hpp"

#include "gradflow/autograd/allocator.hpp"

#include <map>
#include <memory>
#include <mutex>

namespace gradflow {

// Static methods implementation
Device DeviceManager::getDefaultDevice() {
    return cpu();
}

Device DeviceManager::getDevice(DeviceType type, int index) {
    return Device(type, index);
}

bool DeviceManager::isDeviceAvailable(DeviceType type, int index) {
    switch (type) {
        case DeviceType::CPU:
            // CPU is always available
            return index == 0;

        case DeviceType::CUDA:
            // TODO: Implement CUDA availability check
            // For now, return false as CUDA is not yet implemented
            return false;

        case DeviceType::METAL:
            // TODO: Implement Metal availability check
            // For now, return false as Metal is not yet implemented
            return false;

        default:
            return false;
    }
}

int DeviceManager::getDeviceCount(DeviceType type) {
    switch (type) {
        case DeviceType::CPU:
            // Only one CPU device is supported
            return 1;

        case DeviceType::CUDA:
            // TODO: Query CUDA device count
            return 0;

        case DeviceType::METAL:
            // TODO: Query Metal device count
            return 0;

        default:
            return 0;
    }
}

std::shared_ptr<DeviceAllocator> DeviceManager::getAllocator(const Device& device) {
    // Thread-safe allocator cache using a static map
    static std::mutex mutex;
    static std::map<std::pair<DeviceType, int>, std::shared_ptr<DeviceAllocator>> allocator_cache;

    std::lock_guard<std::mutex> lock(mutex);

    // Create a key for the cache
    auto key = std::make_pair(device.type(), device.index());

    // Check if allocator is already cached
    auto it = allocator_cache.find(key);
    if (it != allocator_cache.end()) {
        return it->second;
    }

    // Create a new allocator based on device type
    std::shared_ptr<DeviceAllocator> allocator;

    switch (device.type()) {
        case DeviceType::CPU:
            allocator = std::make_shared<CPUAllocator>();
            break;

        case DeviceType::CUDA:
            // TODO: Create CUDA allocator when implemented
            throw std::runtime_error("CUDA allocator not yet implemented");

        case DeviceType::METAL:
            // TODO: Create Metal allocator when implemented
            throw std::runtime_error("Metal allocator not yet implemented");

        default:
            throw std::runtime_error("Unknown device type");
    }

    // Cache the allocator
    allocator_cache[key] = allocator;

    return allocator;
}

}  // namespace gradflow
