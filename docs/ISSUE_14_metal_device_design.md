# Issue #14: Metal Device と Allocator の設計書

## 1. 調査・リサーチ結果

### Metal API の最新ベストプラクティス

Apple Silicon の Unified Memory Architecture では、CPU と GPU が同一の物理メモリプールを共有します。これにより、従来の CUDA のような明示的なメモリコピーのオーバーヘッドを大幅に削減できます。

#### 重要な設計原則
1. **Storage Mode の選択**: Unified Memory システムでは `MTLResourceStorageModeShared` を使用することで、CPU と GPU の両方から直接アクセス可能になります
2. **MTLDevice の再利用**: 各 GPU につき 1 つの MTLDevice オブジェクトのみを作成し、すべての Metal 操作で再利用します
3. **MTLBuffer の再利用**: 特に静的データの場合、MTLBuffer オブジェクトを可能な限り再利用します。レンダリングや計算ループ内での新規リソース作成は避けます
4. **メモリ使用量の制限**: 実際には物理メモリの約 75% が GPU 使用の推奨最大値です（128 GB システムでは約 96 GB）

### C++ による Metal 実装

Apple は公式に **metal-cpp** を提供しており、Objective-C の Metal API に対する軽量な C++ ラッパーとして利用できます。metal-cpp の特徴：

- **ヘッダーオンリーライブラリ**: 追加のビルド依存なしで統合可能
- **ゼロオーバーヘッド**: インライン関数呼び出しで実装され、Objective-C API への 1 対 1 マッピング
- **100% API カバレッジ**: Metal API のすべての機能を提供
- **手動メモリ管理**: C++ オブジェクトは ARC の対象外のため、retain/release を明示的に管理する必要があります

#### メモリ管理のルール
- `alloc`, `new`, `copy`, `mutableCopy`, `Create` で始まるメソッドは retainCount = 1 のオブジェクトを返します
- 所有権を放棄するには `release()` または `autorelease()` を呼び出します
- metal-cpp は `NS::SharedPtr<T>` 型を提供し、RAII パターンをサポートします
- `NS::TransferPtr()` 関数により、retain count を増やさずに所有権を移譲できます

### 参考文献
- [Metal Best Practices Guide: Resource Options](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/ResourceOptions.html) - リソースオプションとストレージモードの選択
- [Metal Best Practices Guide: Persistent Objects](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/PersistentObjects.html) - MTLDevice と MTLBuffer の再利用戦略
- [MTLBuffer | Apple Developer Documentation](https://developer.apple.com/documentation/metal/mtlbuffer) - 公式 MTLBuffer ドキュメント
- [Getting started with Metal-cpp](https://developer.apple.com/metal/cpp/) - metal-cpp の公式ガイド
- [Program Metal in C++ with metal-cpp - WWDC22](https://developer.apple.com/videos/play/wwdc2022/10160/) - WWDC セッション

---

## 2. 分析と評価

### 現状の課題
- Phase 1-2 では CPU のみをサポートしており、Apple Silicon の GPU 能力を活用できていない
- `DeviceManager` は Metal サポートを想定しているが、実装は未完成（`isDeviceAvailable` が常に false を返す）
- Metal API は Objective-C で提供されており、C++ プロジェクトとの統合に工夫が必要

### 採用すべき設計原則

#### 1. 単一責任の原則 (SRP)
- **MetalAllocator**: メモリ割り当てと解放のみを担当
- **MetalDevice**: デバイスの初期化とコマンドキューの管理
- **MetalBuffer**: MTLBuffer のライフタイム管理（将来的に追加）

#### 2. 開放閉鎖原則 (OCP)
- `DeviceAllocator` インターフェースを拡張することで、既存の CPU コードに影響を与えずに Metal サポートを追加
- 将来的に CUDA Allocator も同様のパターンで追加可能

#### 3. リスコフ置換原則 (LSP)
- `MetalAllocator` は `DeviceAllocator` のすべての契約を満たす必要があります
- `allocate()` が nullptr を返す場合は `std::bad_alloc` をスローする（CPUAllocator と同じ）

#### 4. 依存性逆転の原則 (DIP)
- 高レベルモジュール（Tensor, Storage）は `DeviceAllocator` 抽象インターフェースに依存
- Metal 固有の実装詳細（Objective-C オブジェクト）は .mm ファイル内に隠蔽

#### 5. RAII (Resource Acquisition Is Initialization)
- MTLDevice と MTLBuffer のライフタイムは C++ オブジェクトのスコープと連動
- デストラクタで自動的に Metal リソースを解放

---

## 3. 推奨アーキテクチャ案

### 設計のコンセプト

**3 層アーキテクチャ**:
1. **C++ Public Interface** (`include/gradflow/autograd/metal/device.hpp`): クリーンな C++ API、Objective-C の詳細を隠蔽
2. **Objective-C++ Implementation** (`src/autograd/metal/device.mm`): Metal API の呼び出し
3. **Integration Layer** (`src/autograd/device.cpp`): `DeviceManager` との統合

**主要コンポーネント**:
- `MetalDevice`: MTLDevice と MTLCommandQueue のラッパー
- `MetalAllocator`: MTLBuffer の割り当てと解放を管理
- `MetalBuffer` (内部クラス): MTLBuffer のスマートポインタラッパー

### クラス設計

#### 3.1 MetalDevice クラス

```cpp
// include/gradflow/autograd/metal/device.hpp
#pragma once

#ifdef __APPLE__

#include <cstddef>
#include <memory>
#include <string>

namespace gradflow {
namespace metal {

// Forward declaration (Objective-C オブジェクトを隠蔽)
class MetalDeviceImpl;

/**
 * @brief Metal GPU デバイスのラッパークラス
 *
 * Apple Silicon の Metal GPU デバイスを抽象化します。
 * MTLDevice と MTLCommandQueue のライフタイムを管理します。
 *
 * 使用例:
 * @code
 *   auto device = MetalDevice::create();
 *   if (device) {
 *       std::cout << device->name() << std::endl;
 *   }
 * @endcode
 */
class MetalDevice {
public:
    /**
     * @brief デフォルトの Metal デバイスを作成
     *
     * システムのデフォルト GPU (MTLCreateSystemDefaultDevice) を使用します。
     *
     * @return MetalDevice のユニークポインタ、失敗時は nullptr
     */
    static std::unique_ptr<MetalDevice> create();

    /**
     * @brief Metal デバイスが利用可能かチェック
     * @return Metal が利用可能なら true
     */
    static bool isAvailable();

    /**
     * @brief 利用可能な Metal デバイスの数を取得
     * @return デバイス数 (通常は 0 または 1)
     */
    static int getDeviceCount();

    // コピー・ムーブ禁止 (RAII 管理)
    MetalDevice(const MetalDevice&) = delete;
    MetalDevice& operator=(const MetalDevice&) = delete;
    MetalDevice(MetalDevice&&) = delete;
    MetalDevice& operator=(MetalDevice&&) = delete;

    ~MetalDevice();

    /**
     * @brief デバイス名を取得
     * @return デバイス名 (例: "Apple M1")
     */
    std::string name() const;

    /**
     * @brief 推奨される最大ワーキングセットサイズを取得
     *
     * GPU が効率的に処理できる最大メモリサイズ（バイト）。
     * 通常は物理メモリの約 75%。
     *
     * @return 推奨最大メモリサイズ (bytes)
     */
    size_t recommendedMaxWorkingSetSize() const;

    /**
     * @brief Unified Memory をサポートしているかチェック
     * @return Apple Silicon なら常に true
     */
    bool hasUnifiedMemory() const;

    /**
     * @brief 内部実装へのアクセス (Allocator 用)
     * @return 内部実装ポインタ
     */
    MetalDeviceImpl* impl() const { return impl_.get(); }

private:
    MetalDevice();  // Private constructor (factory pattern)

    std::unique_ptr<MetalDeviceImpl> impl_;
};

}  // namespace metal
}  // namespace gradflow

#endif  // __APPLE__
```

#### 3.2 MetalAllocator クラス

```cpp
// include/gradflow/autograd/metal/allocator.hpp
#pragma once

#ifdef __APPLE__

#include "gradflow/autograd/allocator.hpp"
#include "gradflow/autograd/metal/device.hpp"

#include <cstddef>
#include <memory>

namespace gradflow {
namespace metal {

/**
 * @brief Metal GPU メモリアロケータ
 *
 * MTLBuffer を使用して GPU メモリを割り当てます。
 * Unified Memory Architecture により、CPU からも直接アクセス可能です。
 *
 * メモリモード:
 * - MTLResourceStorageModeShared: CPU/GPU 両方からアクセス可能
 * - Coherent: 自動的に同期される
 *
 * 使用例:
 * @code
 *   auto device = MetalDevice::create();
 *   MetalAllocator allocator(device.get());
 *   void* ptr = allocator.allocate(1024);
 *   // ... GPU/CPU から使用 ...
 *   allocator.deallocate(ptr);
 * @endcode
 */
class MetalAllocator : public DeviceAllocator {
public:
    /**
     * @brief デフォルトアライメント (256 bytes)
     *
     * Metal では 256 バイトアライメントが推奨されます。
     */
    static constexpr size_t kDefaultAlignment = 256;

    /**
     * @brief MetalAllocator を構築
     *
     * @param device Metal デバイス (非 null)
     * @param alignment アライメント要件 (デフォルト: 256 bytes)
     * @throws std::invalid_argument device が null の場合
     */
    explicit MetalAllocator(MetalDevice* device, size_t alignment = kDefaultAlignment);

    ~MetalAllocator() override;

    // コピー・ムーブ禁止
    MetalAllocator(const MetalAllocator&) = delete;
    MetalAllocator& operator=(const MetalAllocator&) = delete;
    MetalAllocator(MetalAllocator&&) = delete;
    MetalAllocator& operator=(MetalAllocator&&) = delete;

    /**
     * @brief GPU メモリを割り当て
     *
     * MTLBuffer を作成し、Shared storage mode で割り当てます。
     * Unified Memory により、CPU からも直接アクセス可能です。
     *
     * @param size 割り当てサイズ (bytes)
     * @return 割り当てられたメモリへのポインタ
     * @throws std::bad_alloc 割り当て失敗時
     */
    void* allocate(size_t size) override;

    /**
     * @brief GPU メモリを解放
     *
     * @param ptr 解放するメモリポインタ
     */
    void deallocate(void* ptr) override;

    /**
     * @brief Metal デバイスを取得
     * @return Device オブジェクト
     */
    Device device() const override;

    /**
     * @brief アライメント要件を取得
     * @return アライメント (bytes)
     */
    size_t alignment() const override { return alignment_; }

    /**
     * @brief CPU から GPU メモリにデータをコピー
     *
     * Unified Memory では通常の memcpy と同等ですが、
     * 将来的に Private storage mode をサポートする場合に備えて提供します。
     *
     * @param dst GPU メモリポインタ
     * @param src CPU メモリポインタ
     * @param size コピーサイズ (bytes)
     */
    void copyFromCPU(void* dst, const void* src, size_t size);

    /**
     * @brief GPU から CPU メモリにデータをコピー
     *
     * @param dst CPU メモリポインタ
     * @param src GPU メモリポインタ
     * @param size コピーサイズ (bytes)
     */
    void copyToCPU(void* dst, const void* src, size_t size);

    /**
     * @brief GPU 操作を同期
     *
     * 保留中のすべてのコマンドが完了するまで待機します。
     */
    void synchronize();

private:
    MetalDevice* device_;
    size_t alignment_;

    // MTLBuffer の管理用内部データ構造
    struct BufferInfo;
    std::unique_ptr<std::unordered_map<void*, BufferInfo>> buffer_map_;
};

/**
 * @brief デフォルトの Metal Allocator を取得
 *
 * シングルトンパターンで MetalAllocator を管理します。
 *
 * @return Shared pointer to MetalAllocator, Metal が利用不可なら nullptr
 */
std::shared_ptr<DeviceAllocator> getDefaultMetalAllocator();

}  // namespace metal
}  // namespace gradflow

#endif  // __APPLE__
```

---

## 4. 実装の詳細

### 4.1 Objective-C++ 実装 (device.mm)

```cpp
// src/autograd/metal/device.mm
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "gradflow/autograd/metal/device.hpp"
#include <stdexcept>

namespace gradflow {
namespace metal {

// Objective-C オブジェクトを保持する内部実装
class MetalDeviceImpl {
public:
    id<MTLDevice> device;
    id<MTLCommandQueue> command_queue;

    MetalDeviceImpl(id<MTLDevice> dev) : device([dev retain]) {
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
};

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
    return impl_->device.recommendedMaxWorkingSetSize;
}

bool MetalDevice::hasUnifiedMemory() const {
    return impl_->device.hasUnifiedMemory;
}

}  // namespace metal
}  // namespace gradflow
```

### 4.2 MetalAllocator の実装

```cpp
// src/autograd/metal/allocator.mm
#import <Metal/Metal.h>

#include "gradflow/autograd/metal/allocator.hpp"
#include "gradflow/autograd/metal/device.hpp"
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace gradflow {
namespace metal {

// MTLBuffer の情報を保持
struct MetalAllocator::BufferInfo {
    id<MTLBuffer> buffer;
    size_t size;

    BufferInfo(id<MTLBuffer> buf, size_t sz) : buffer([buf retain]), size(sz) {}

    ~BufferInfo() {
        [buffer release];
    }
};

MetalAllocator::MetalAllocator(MetalDevice* device, size_t alignment)
    : device_(device), alignment_(alignment) {
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
        auto* impl = device_->impl();
        id<MTLDevice> mtl_device = impl->device;

        // Shared storage mode: CPU と GPU 両方からアクセス可能
        id<MTLBuffer> buffer = [mtl_device
            newBufferWithLength:aligned_size
            options:MTLResourceStorageModeShared];

        if (!buffer) {
            throw std::bad_alloc();
        }

        void* ptr = buffer.contents;

        // BufferInfo を登録
        buffer_map_->emplace(ptr, BufferInfo(buffer, aligned_size));

        [buffer release];  // BufferInfo が retain しているので release
        return ptr;
    }
}

void MetalAllocator::deallocate(void* ptr) {
    if (!ptr) {
        return;
    }

    auto it = buffer_map_->find(ptr);
    if (it == buffer_map_->end()) {
        throw std::runtime_error("Attempting to deallocate unknown pointer");
    }

    buffer_map_->erase(it);
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
        auto* impl = device_->impl();
        id<MTLCommandBuffer> cmd_buffer = [impl->command_queue commandBuffer];
        [cmd_buffer commit];
        [cmd_buffer waitUntilCompleted];
    }
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

}  // namespace metal
}  // namespace gradflow
```

---

## 5. DeviceManager との統合

### device.cpp への追加

```cpp
// src/autograd/device.cpp に追加

#ifdef __APPLE__
#include "gradflow/autograd/metal/device.hpp"
#include "gradflow/autograd/metal/allocator.hpp"
#endif

bool DeviceManager::isDeviceAvailable(DeviceType type, int index) {
    switch (type) {
        case DeviceType::CPU:
            return index == 0;

        case DeviceType::CUDA:
            // TODO: Implement CUDA availability check
            return false;

        case DeviceType::METAL:
#ifdef __APPLE__
            return index == 0 && metal::MetalDevice::isAvailable();
#else
            return false;
#endif

        default:
            return false;
    }
}

int DeviceManager::getDeviceCount(DeviceType type) {
    switch (type) {
        case DeviceType::CPU:
            return 1;

        case DeviceType::CUDA:
            // TODO: Query CUDA device count
            return 0;

        case DeviceType::METAL:
#ifdef __APPLE__
            return metal::MetalDevice::getDeviceCount();
#else
            return 0;
#endif

        default:
            return 0;
    }
}

std::shared_ptr<DeviceAllocator> DeviceManager::getAllocator(const Device& device) {
    // ... 既存のコード ...

    switch (device.type()) {
        case DeviceType::CPU:
            allocator = std::make_shared<CPUAllocator>();
            break;

        case DeviceType::CUDA:
            throw std::runtime_error("CUDA allocator not yet implemented");

        case DeviceType::METAL:
#ifdef __APPLE__
            allocator = metal::getDefaultMetalAllocator();
            if (!allocator) {
                throw std::runtime_error("Metal device not available");
            }
#else
            throw std::runtime_error("Metal is only available on Apple platforms");
#endif
            break;

        default:
            throw std::runtime_error("Unknown device type");
    }

    // ... 既存のコード ...
}
```

---

## 6. CMake 統合

### CMakeLists.txt への追加

```cmake
# src/autograd/CMakeLists.txt

# 既存の gradflow_impl ライブラリ
add_library(gradflow_impl STATIC
    device.cpp
    # ... 他のソースファイル ...
)

# Metal サポート (macOS のみ)
if(APPLE AND GRADFLOW_ENABLE_METAL)
    message(STATUS "Enabling Metal GPU backend")

    # Metal ソースを追加
    target_sources(gradflow_impl PRIVATE
        metal/device.mm
        metal/allocator.mm
    )

    # Metal framework をリンク
    target_link_libraries(gradflow_impl PUBLIC
        "-framework Foundation"
        "-framework Metal"
    )

    # Metal を有効化するマクロを定義
    target_compile_definitions(gradflow_impl PUBLIC GRADFLOW_HAS_METAL)

    # Objective-C++ フラグ
    set_source_files_properties(
        metal/device.mm
        metal/allocator.mm
        PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"  # ARC を有効化
    )
endif()
```

---

## 7. テスト設計

### test_metal_device.cpp

```cpp
// tests/test_metal_device.cpp
#include <gtest/gtest.h>

#ifdef __APPLE__
#include "gradflow/autograd/metal/device.hpp"
#include "gradflow/autograd/metal/allocator.hpp"
#include <cstring>
#include <vector>

using namespace gradflow;
using namespace gradflow::metal;

class MetalDeviceTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!MetalDevice::isAvailable()) {
            GTEST_SKIP() << "Metal is not available on this system";
        }

        device_ = MetalDevice::create();
        ASSERT_NE(device_, nullptr) << "Failed to create Metal device";

        allocator_ = std::make_unique<MetalAllocator>(device_.get());
    }

    std::unique_ptr<MetalDevice> device_;
    std::unique_ptr<MetalAllocator> allocator_;
};

// Test 1: デバイス情報の取得
TEST_F(MetalDeviceTest, DeviceInfo) {
    EXPECT_FALSE(device_->name().empty());
    EXPECT_GT(device_->recommendedMaxWorkingSetSize(), 0);
    EXPECT_TRUE(device_->hasUnifiedMemory());

    std::cout << "Device: " << device_->name() << std::endl;
    std::cout << "Max working set: "
              << device_->recommendedMaxWorkingSetSize() / (1024 * 1024 * 1024)
              << " GB" << std::endl;
}

// Test 2: メモリ割り当てと解放
TEST_F(MetalDeviceTest, Allocation) {
    // 小さいサイズ
    void* ptr1 = allocator_->allocate(1024);
    ASSERT_NE(ptr1, nullptr);
    allocator_->deallocate(ptr1);

    // 大きいサイズ (1 MB)
    void* ptr2 = allocator_->allocate(1024 * 1024);
    ASSERT_NE(ptr2, nullptr);
    allocator_->deallocate(ptr2);

    // 0 バイト
    void* ptr3 = allocator_->allocate(0);
    EXPECT_EQ(ptr3, nullptr);
}

// Test 3: CPU ↔ GPU メモリコピー
TEST_F(MetalDeviceTest, MemoryCopy) {
    constexpr size_t size = 1024;

    // CPU 側のデータ
    std::vector<float> cpu_data(size);
    for (size_t i = 0; i < size; ++i) {
        cpu_data[i] = static_cast<float>(i);
    }

    // GPU メモリを割り当て
    void* gpu_ptr = allocator_->allocate(size * sizeof(float));
    ASSERT_NE(gpu_ptr, nullptr);

    // CPU → GPU
    allocator_->copyFromCPU(gpu_ptr, cpu_data.data(), size * sizeof(float));

    // GPU → CPU (別のバッファに)
    std::vector<float> cpu_result(size);
    allocator_->copyToCPU(cpu_result.data(), gpu_ptr, size * sizeof(float));

    // 検証
    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(cpu_result[i], cpu_data[i]);
    }

    allocator_->deallocate(gpu_ptr);
}

// Test 4: Unified Memory による直接アクセス
TEST_F(MetalDeviceTest, UnifiedMemoryAccess) {
    constexpr size_t size = 256;

    // GPU メモリを割り当て
    float* gpu_ptr = static_cast<float*>(allocator_->allocate(size * sizeof(float)));
    ASSERT_NE(gpu_ptr, nullptr);

    // CPU から直接書き込み (Unified Memory)
    for (size_t i = 0; i < size; ++i) {
        gpu_ptr[i] = static_cast<float>(i * 2);
    }

    // CPU から直接読み込み
    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(gpu_ptr[i], static_cast<float>(i * 2));
    }

    allocator_->deallocate(gpu_ptr);
}

// Test 5: 複数のバッファを同時に管理
TEST_F(MetalDeviceTest, MultipleBuffers) {
    constexpr int num_buffers = 10;
    std::vector<void*> buffers;

    // 割り当て
    for (int i = 0; i < num_buffers; ++i) {
        void* ptr = allocator_->allocate((i + 1) * 1024);
        ASSERT_NE(ptr, nullptr);
        buffers.push_back(ptr);
    }

    // 解放
    for (void* ptr : buffers) {
        allocator_->deallocate(ptr);
    }
}

// Test 6: DeviceManager との統合
TEST(MetalDeviceManagerTest, Integration) {
    if (!MetalDevice::isAvailable()) {
        GTEST_SKIP() << "Metal is not available";
    }

    // デバイスの可用性
    EXPECT_TRUE(DeviceManager::isDeviceAvailable(DeviceType::METAL, 0));
    EXPECT_EQ(DeviceManager::getDeviceCount(DeviceType::METAL), 1);

    // Allocator の取得
    Device metal_device = metal(0);
    auto allocator = DeviceManager::getAllocator(metal_device);
    ASSERT_NE(allocator, nullptr);
    EXPECT_EQ(allocator->device().type(), DeviceType::METAL);

    // 割り当てテスト
    void* ptr = allocator->allocate(1024);
    ASSERT_NE(ptr, nullptr);
    allocator->deallocate(ptr);
}

#endif  // __APPLE__
```

### テストの実行

```bash
# Metal サポートを有効化してビルド
cmake -S . -B build -DGRADFLOW_ENABLE_METAL=ON -DGRADFLOW_BUILD_TESTS=ON
cmake --build build

# テスト実行
cd build
ctest -R MetalDevice -V
```

---

## 8. エラーハンドリング

### エラーケースと対応

| エラーケース | 処理 | 例外 |
|-------------|------|------|
| Metal デバイスが存在しない | `MetalDevice::create()` が `nullptr` を返す | なし |
| Metal が利用できない環境 | `DeviceManager::isDeviceAvailable()` が `false` | なし |
| メモリ割り当て失敗 | `std::bad_alloc` をスロー | `std::bad_alloc` |
| コマンドキュー作成失敗 | `std::runtime_error` をスロー | `std::runtime_error` |
| 不正なポインタの解放 | `std::runtime_error` をスロー | `std::runtime_error` |
| null デバイスで Allocator 作成 | `std::invalid_argument` をスロー | `std::invalid_argument` |

### エラーハンドリングの例

```cpp
try {
    auto device = MetalDevice::create();
    if (!device) {
        std::cerr << "Metal is not available, falling back to CPU" << std::endl;
        // CPU にフォールバック
        auto cpu_allocator = getDefaultCpuAllocator();
    } else {
        MetalAllocator allocator(device.get());
        void* ptr = allocator.allocate(1024 * 1024);
        // ... 使用 ...
        allocator.deallocate(ptr);
    }
} catch (const std::bad_alloc& e) {
    std::cerr << "Out of GPU memory: " << e.what() << std::endl;
} catch (const std::runtime_error& e) {
    std::cerr << "Metal runtime error: " << e.what() << std::endl;
}
```

---

## 9. トレードオフと設計判断

### メリット

1. **Unified Memory の活用**: 明示的なメモリコピーが不要で、CPU と GPU 間のデータ転送オーバーヘッドが最小
2. **クリーンな抽象化**: Objective-C の詳細を完全に隠蔽し、C++ コードから透過的に使用可能
3. **RAII によるメモリ安全性**: C++ のスコープ管理により、Metal リソースのリークを防止
4. **既存コードとの統合**: `DeviceAllocator` インターフェースを実装することで、既存の Tensor/Storage コードをそのまま使用可能
5. **テスタビリティ**: Metal が利用できない環境でもコンパイル可能（条件付きコンパイル）

### リスク・注意点

1. **プラットフォーム依存**: macOS/iOS 専用で、クロスプラットフォームビルドに条件分岐が必要
2. **メモリ管理の複雑性**: Objective-C の ARC と C++ の手動メモリ管理の混在に注意が必要
3. **パフォーマンス**: Shared storage mode は Private mode より遅い可能性（ただし、データ転送コストを考慮すると総合的に高速）
4. **テストの制限**: CI 環境が macOS である必要があり、GitHub Actions のコストが高い

---

## 10. 次のステップ (github-issue-implementer への指示)

### 実装タスクリスト

#### Phase 1: ヘッダーファイルの作成
1. `include/gradflow/autograd/metal/device.hpp` を作成
2. `include/gradflow/autograd/metal/allocator.hpp` を作成
3. Forward declaration と PIMPL パターンを使用して Objective-C の詳細を隠蔽

#### Phase 2: Objective-C++ 実装
1. `src/autograd/metal/device.mm` を作成
   - `MetalDeviceImpl` クラス (MTLDevice と MTLCommandQueue を保持)
   - `MetalDevice::create()`, `isAvailable()`, `getDeviceCount()` の実装
2. `src/autograd/metal/allocator.mm` を作成
   - `MetalAllocator::allocate()`, `deallocate()` の実装
   - MTLBuffer の管理 (BufferInfo 構造体)

#### Phase 3: DeviceManager との統合
1. `src/autograd/device.cpp` を更新
   - `isDeviceAvailable()` に Metal サポートを追加
   - `getDeviceCount()` に Metal サポートを追加
   - `getAllocator()` に Metal サポートを追加

#### Phase 4: CMake 設定
1. `src/autograd/CMakeLists.txt` を更新
   - Metal ソースファイルの追加 (GRADFLOW_ENABLE_METAL フラグ)
   - Metal framework のリンク
   - Objective-C++ フラグ (-fobjc-arc)

#### Phase 5: テストの実装
1. `tests/test_metal_device.cpp` を作成
   - MetalDeviceTest::DeviceInfo
   - MetalDeviceTest::Allocation
   - MetalDeviceTest::MemoryCopy
   - MetalDeviceTest::UnifiedMemoryAccess
   - MetalDeviceTest::MultipleBuffers
   - MetalDeviceManagerTest::Integration

#### Phase 6: ドキュメント
1. README.md に Metal サポートの説明を追加
2. ビルド手順を更新 (GRADFLOW_ENABLE_METAL オプション)

### 完了基準
- [ ] すべてのテストが pass
- [ ] Metal が利用できない環境でもビルド可能
- [ ] GPU メモリの割り当てと解放が正常に動作
- [ ] CPU ↔ GPU のデータ転送が動作
- [ ] Unified Memory による直接アクセスが動作
- [ ] DeviceManager 経由で MetalAllocator を取得可能
- [ ] CI で Metal テストが実行される (macOS runner)

### 推奨実装順序
1. ヘッダーファイル → コンパイルエラーがないことを確認
2. device.mm の実装 → デバイス作成テストを実行
3. allocator.mm の実装 → メモリ割り当てテストを実行
4. DeviceManager 統合 → 統合テストを実行
5. 包括的なテストの追加

---

## 11. 参考資料とリンク

### Apple 公式ドキュメント
- [Metal Best Practices Guide: Resource Options](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/ResourceOptions.html)
- [Metal Best Practices Guide: Persistent Objects](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/PersistentObjects.html)
- [MTLBuffer Documentation](https://developer.apple.com/documentation/metal/mtlbuffer)
- [Getting started with Metal-cpp](https://developer.apple.com/metal/cpp/)
- [Program Metal in C++ with metal-cpp - WWDC22](https://developer.apple.com/videos/play/wwdc2022/10160/)

### 技術ブログ
- [Working with memory in Metal](https://metalkit.org/2017/04/30/working-with-memory-in-metal/)
- [Working with memory in Metal part 2](https://metalkit.org/working-with-memory-in-metal-part-2/)

### オープンソース実装例
- [GitHub - philipturner/metal-usm](https://github.com/philipturner/metal-usm) - CPU ポインタから GPU アクセス
- [GitHub - BoltzmannEntropy/metalQwen3](https://github.com/BoltzmannEntropy/metalQwen3) - Metal による Transformer 実装

---

以上が Issue #14 の詳細設計書です。この設計に基づき、github-issue-implementer が実装を進めることで、GradFlow に Metal GPU サポートが追加されます。
