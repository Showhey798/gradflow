#include <gtest/gtest.h>

#ifdef __APPLE__
#include "gradflow/autograd/device.hpp"
#include "gradflow/autograd/metal/allocator.hpp"
#include "gradflow/autograd/metal/device.hpp"

#include <cstring>
#include <vector>

using namespace gradflow;
using namespace gradflow::gpu;

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
              << device_->recommendedMaxWorkingSetSize() / (1024 * 1024 * 1024) << " GB"
              << std::endl;
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
    auto* gpu_ptr = static_cast<float*>(allocator_->allocate(size * sizeof(float)));
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
        void* ptr = allocator_->allocate(static_cast<size_t>(i + 1) * 1024);
        ASSERT_NE(ptr, nullptr);
        buffers.push_back(ptr);
    }

    // 解放
    for (void* ptr : buffers) {
        allocator_->deallocate(ptr);
    }
}

// Test 6: デバイスの可用性確認
TEST(MetalAvailability, IsAvailable) {
    // このテストは Metal が利用できない環境でも実行される
    if (MetalDevice::isAvailable()) {
        EXPECT_EQ(MetalDevice::getDeviceCount(), 1);

        auto device = MetalDevice::create();
        ASSERT_NE(device, nullptr);
        EXPECT_FALSE(device->name().empty());
    } else {
        EXPECT_EQ(MetalDevice::getDeviceCount(), 0);
    }
}

// Test 7: DeviceManager との統合
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

// Test 8: Allocator のアライメント
TEST_F(MetalDeviceTest, Alignment) {
    EXPECT_EQ(allocator_->alignment(), MetalAllocator::kDefaultAlignment);

    // 割り当てられたメモリがアライメント要件を満たしているか
    void* ptr = allocator_->allocate(100);
    ASSERT_NE(ptr, nullptr);

    auto addr = reinterpret_cast<uintptr_t>(ptr);
    EXPECT_EQ(addr % MetalAllocator::kDefaultAlignment, 0) << "Memory is not properly aligned";

    allocator_->deallocate(ptr);
}

// Test 9: エラーハンドリング - 不正なポインタの解放
TEST_F(MetalDeviceTest, InvalidDeallocate) {
    int dummy = 0;
    void* invalid_ptr = &dummy;

    // 不正なポインタはサイレントに無視される（std::free と同様の動作）
    EXPECT_NO_THROW(allocator_->deallocate(invalid_ptr));

    // nullptr も許容される
    EXPECT_NO_THROW(allocator_->deallocate(nullptr));
}

// Test 10: 同期
TEST_F(MetalDeviceTest, Synchronize) {
    void* ptr = allocator_->allocate(1024);
    ASSERT_NE(ptr, nullptr);

    // 同期を実行 (エラーが発生しないことを確認)
    EXPECT_NO_THROW(allocator_->synchronize());

    allocator_->deallocate(ptr);
}

#endif  // __APPLE__

// Metal が利用できない環境でもテストが実行されることを確認
TEST(MetalStub, NonApplePlatform) {
#ifndef __APPLE__
    EXPECT_FALSE(DeviceManager::isDeviceAvailable(DeviceType::METAL, 0));
    EXPECT_EQ(DeviceManager::getDeviceCount(DeviceType::METAL), 0);

    Device metal_device = metal(0);
    EXPECT_THROW(DeviceManager::getAllocator(metal_device), std::runtime_error);
#else
    GTEST_SKIP() << "This test is for non-Apple platforms";
#endif
}
