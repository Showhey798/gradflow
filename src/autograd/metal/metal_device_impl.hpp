#pragma once

#ifdef __APPLE__

#import <Metal/Metal.h>

#include <stdexcept>

namespace gradflow {
namespace gpu {

// Objective-C オブジェクトを保持する内部実装
// この定義を .mm ファイル間で共有する
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

}  // namespace gpu
}  // namespace gradflow

#endif  // __APPLE__
