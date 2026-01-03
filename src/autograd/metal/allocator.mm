#include "gradflow/autograd/metal/allocator.hpp"

#include "gradflow/autograd/metal/device.hpp"

#import <Metal/Metal.h>

#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace gradflow {
namespace gpu {

// MTLBuffer の情報を保持
struct MetalAllocator::BufferInfo {
    id<MTLBuffer> buffer;
    size_t size;

    BufferInfo(id<MTLBuffer> buf, size_t sz) : buffer([buf retain]), size(sz) {}

    ~BufferInfo() { [buffer release]; }

    // コピー・ムーブ禁止
    BufferInfo(const BufferInfo&) = delete;
    BufferInfo& operator=(const BufferInfo&) = delete;
    BufferInfo(BufferInfo&&) = delete;
    BufferInfo& operator=(BufferInfo&&) = delete;
};

MetalAllocator::MetalAllocator(MetalDevice* device, size_t alignment)
    : device_(device),
      alignment_(alignment) {
    if (!device_) {
        throw std::invalid_argument("MetalDevice cannot be null");
    }
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("Alignment must be a power of 2");
    }

    buffer_map_ = std::make_unique<std::unordered_map<void*, BufferInfo>>();
}

MetalAllocator::~MetalAllocator() = default;

void* MetalAllocator::allocate(size_t size) {
    if (size == 0) {
        return nullptr;
    }

    // サイズをアライメントに丸める
    size_t aligned_size = (size + alignment_ - 1) & ~(alignment_ - 1);

    @autoreleasepool {
        id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device_->getMetalDevice();

        // Shared storage mode: CPU と GPU 両方からアクセス可能
        id<MTLBuffer> buffer = [mtl_device newBufferWithLength:aligned_size
                                                       options:MTLResourceStorageModeShared];

        if (!buffer) {
            throw std::bad_alloc();
        }

        void* ptr = buffer.contents;

        // BufferInfo を登録
        buffer_map_->emplace(std::piecewise_construct,
                             std::forward_as_tuple(ptr),
                             std::forward_as_tuple(buffer, aligned_size));

        [buffer release];  // BufferInfo が retain しているので release
        return ptr;
    }
}

void MetalAllocator::deallocate(void* ptr) {
    if (!ptr) {
        return;  // nullptr は許容（std::free と同様の動作）
    }

    auto it = buffer_map_->find(ptr);
    if (it == buffer_map_->end()) {
        // 不正なポインタの場合は何もしない（サイレントに無視）
        // Metal buffers は BufferInfo の RAII により自動的に解放されます
        return;
    }

    buffer_map_->erase(it);  // BufferInfo のデストラクタで [buffer release] が呼ばれる
}

Device MetalAllocator::device() const {
    return Device(DeviceType::METAL, 0);
}

void MetalAllocator::copyFromCPU(void* dst, const void* src, size_t size) {
    // Unified Memory なので通常の memcpy で OK
    std::memcpy(dst, src, size);
}

void MetalAllocator::copyToCPU(void* dst, const void* src, size_t size) {
    // Unified Memory なので通常の memcpy で OK
    std::memcpy(dst, src, size);
}

void MetalAllocator::synchronize() {
    // コマンドキューのすべてのコマンドバッファを同期
    @autoreleasepool {
        id<MTLCommandQueue> command_queue =
            (__bridge id<MTLCommandQueue>)device_->getMetalCommandQueue();
        id<MTLCommandBuffer> cmd_buffer = [command_queue commandBuffer];
        [cmd_buffer commit];
        [cmd_buffer waitUntilCompleted];
    }
}

void* MetalAllocator::getBuffer(void* ptr) {
    if (!ptr) {
        return nullptr;
    }

    auto it = buffer_map_->find(ptr);
    if (it == buffer_map_->end()) {
        return nullptr;
    }

    // MTLBuffer を void* として返す（使用側で __bridge でキャスト）
    return (__bridge void*)it->second.buffer;
}

// シングルトン
std::shared_ptr<DeviceAllocator> getDefaultMetalAllocator() {
    static std::shared_ptr<DeviceAllocator> allocator = []() -> std::shared_ptr<DeviceAllocator> {
        auto device = MetalDevice::create();
        if (!device) {
            return nullptr;
        }

        static auto static_device = std::move(device);
        return std::make_shared<MetalAllocator>(static_device.get());
    }();

    return allocator;
}

}  // namespace gpu
}  // namespace gradflow
