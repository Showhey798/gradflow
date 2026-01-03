#include "gradflow/autograd/metal/device.hpp"

#import <Metal/Metal.h>

#include <stdexcept>

#import <Foundation/Foundation.h>

namespace gradflow {
namespace gpu {

// Objective-C オブジェクトを保持する内部実装
// PIMPL パターンにより、この定義は .mm ファイル内でのみ公開される
class MetalDeviceImpl {
public:
    id<MTLDevice> device;
    id<MTLCommandQueue> command_queue;

    explicit MetalDeviceImpl(id<MTLDevice> dev) : device([dev retain]) {
        command_queue = [device newCommandQueue];
        if (!command_queue) {
            [device release];
            throw std::runtime_error("Failed to create Metal command queue");
        }
    }

    ~MetalDeviceImpl() {
        [command_queue release];
        [device release];
    }

    // コピー・ムーブ禁止
    MetalDeviceImpl(const MetalDeviceImpl&) = delete;
    MetalDeviceImpl& operator=(const MetalDeviceImpl&) = delete;
    MetalDeviceImpl(MetalDeviceImpl&&) = delete;
    MetalDeviceImpl& operator=(MetalDeviceImpl&&) = delete;
};

// MetalDevice の実装
MetalDevice::MetalDevice() : impl_(nullptr) {}

MetalDevice::~MetalDevice() = default;

std::unique_ptr<MetalDevice> MetalDevice::create() {
    // MTLCreateSystemDefaultDevice は Create ルールで所有権を持つオブジェクトを返す
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

bool MetalDevice::isAvailable() {
    // MTLCreateSystemDefaultDevice は Create ルールで所有権を持つオブジェクトを返す
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        return false;
    }
    [device release];
    return true;
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

void* MetalDevice::getMetalDevice() const {
    return (__bridge void*)impl_->device;
}

void* MetalDevice::getMetalCommandQueue() const {
    return (__bridge void*)impl_->command_queue;
}

}  // namespace gpu
}  // namespace gradflow
