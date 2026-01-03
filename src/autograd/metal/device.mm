#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "gradflow/autograd/metal/device.hpp"
#include "metal_device_impl.hpp"

#include <stdexcept>

namespace gradflow {
namespace gpu {

// MetalDevice の実装
MetalDevice::MetalDevice() : impl_(nullptr) {}

MetalDevice::~MetalDevice() = default;

std::unique_ptr<MetalDevice> MetalDevice::create() {
    @autoreleasepool {
        id<MTLDevice> mtl_device = MTLCreateSystemDefaultDevice();
        if (!mtl_device) {
            return nullptr;
        }

        auto device = std::unique_ptr<MetalDevice>(new MetalDevice());
        try {
            device->impl_ = std::make_unique<MetalDeviceImpl>(mtl_device);
        } catch (...) {
            [mtl_device release];
            return nullptr;
        }

        [mtl_device release];  // impl が retain しているので release
        return device;
    }
}

bool MetalDevice::isAvailable() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        bool available = (device != nil);
        [device release];
        return available;
    }
}

int MetalDevice::getDeviceCount() {
    return isAvailable() ? 1 : 0;
}

std::string MetalDevice::name() const {
    @autoreleasepool {
        NSString* device_name = impl_->device.name;
        return std::string([device_name UTF8String]);
    }
}

size_t MetalDevice::recommendedMaxWorkingSetSize() const {
    return static_cast<size_t>(impl_->device.recommendedMaxWorkingSetSize);
}

bool MetalDevice::hasUnifiedMemory() const {
    return impl_->device.hasUnifiedMemory;
}

}  // namespace gpu
}  // namespace gradflow
