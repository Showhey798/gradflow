#include <gtest/gtest.h>

#include "gradflow/autograd/allocator.hpp"
#include "gradflow/autograd/device.hpp"

namespace gradflow {
namespace {

// DeviceTest: Device クラスの基本機能テスト
class DeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

// CPUDevice の基本機能テスト
TEST_F(DeviceTest, CpuDeviceBasics) {
  Device cpu_device = cpu();

  EXPECT_EQ(cpu_device.type(), DeviceType::CPU);
  EXPECT_EQ(cpu_device.index(), 0);
  EXPECT_TRUE(cpu_device.isCpu());
  EXPECT_FALSE(cpu_device.isCuda());
  EXPECT_FALSE(cpu_device.isMetal());
}

// Device の文字列表現テスト
TEST_F(DeviceTest, DeviceToString) {
  Device cpu_device = cpu();
  EXPECT_EQ(cpu_device.toString(), "cpu:0");

  Device cuda_device = cuda(1);
  EXPECT_EQ(cuda_device.toString(), "cuda:1");

  Device metal_device = metal(0);
  EXPECT_EQ(metal_device.toString(), "metal:0");
}

// Device の比較演算子テスト
TEST_F(DeviceTest, DeviceComparison) {
  Device cpu1 = cpu();
  Device cpu2 = cpu();
  Device cuda1 = cuda(0);
  Device cuda2 = cuda(1);

  EXPECT_EQ(cpu1, cpu2);
  EXPECT_NE(cpu1, cuda1);
  EXPECT_NE(cuda1, cuda2);
}

// DeviceAllocator のテスト
class AllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

// CPUAllocator の基本機能テスト
TEST_F(AllocatorTest, CpuAllocatorBasics) {
  CPUAllocator allocator;

  EXPECT_EQ(allocator.device().type(), DeviceType::CPU);
  EXPECT_EQ(allocator.alignment(), CPUAllocator::kDefaultAlignment);
}

// CPUAllocator のメモリ割り当てテスト
TEST_F(AllocatorTest, CpuAllocatorAllocation) {
  CPUAllocator allocator;

  // 小さいサイズの割り当て
  void* ptr1 = allocator.allocate(100);
  EXPECT_NE(ptr1, nullptr);
  allocator.deallocate(ptr1);

  // 大きいサイズの割り当て
  void* ptr2 = allocator.allocate(1024 * 1024);  // 1MB
  EXPECT_NE(ptr2, nullptr);
  allocator.deallocate(ptr2);

  // ゼロサイズの割り当て
  void* ptr3 = allocator.allocate(0);
  EXPECT_EQ(ptr3, nullptr);
}

// CPUAllocator のメモリアライメントテスト
TEST_F(AllocatorTest, CpuAllocatorAlignment) {
  CPUAllocator allocator(64);

  void* ptr = allocator.allocate(100);
  EXPECT_NE(ptr, nullptr);

  // アライメントチェック
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);

  allocator.deallocate(ptr);
}

// CPUAllocator の無効なアライメント
TEST_F(AllocatorTest, CpuAllocatorInvalidAlignment) {
  // アライメントが 2 のべき乗でない場合は例外
  EXPECT_THROW(CPUAllocator(3), std::invalid_argument);
  EXPECT_THROW(CPUAllocator(0), std::invalid_argument);
}

// デフォルト CPU Allocator のテスト
TEST_F(AllocatorTest, DefaultCpuAllocator) {
  auto allocator = getDefaultCpuAllocator();
  EXPECT_NE(allocator, nullptr);
  EXPECT_EQ(allocator->device().type(), DeviceType::CPU);

  // シングルトンであることを確認
  auto allocator2 = getDefaultCpuAllocator();
  EXPECT_EQ(allocator, allocator2);
}

// 複数回の割り当てと解放のテスト
TEST_F(AllocatorTest, MultipleAllocations) {
  CPUAllocator allocator;

  const size_t kNumAllocations = 100;
  std::vector<void*> ptrs;

  // 複数の割り当て
  for (size_t i = 0; i < kNumAllocations; ++i) {
    void* ptr = allocator.allocate(1024);
    EXPECT_NE(ptr, nullptr);
    ptrs.push_back(ptr);
  }

  // すべて解放
  for (void* ptr : ptrs) {
    allocator.deallocate(ptr);
  }
}

// nullptr の解放テスト
TEST_F(AllocatorTest, DeallocateNullptr) {
  CPUAllocator allocator;
  // nullptr を解放してもクラッシュしないことを確認
  EXPECT_NO_THROW(allocator.deallocate(nullptr));
}

// DeviceManagerTest: デバイスの取得・管理機能のテスト
class DeviceManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

// デフォルト CPU デバイスの取得テスト
TEST_F(DeviceManagerTest, GetDefaultCpuDevice) {
  Device cpu_device = DeviceManager::getDefaultDevice();
  EXPECT_EQ(cpu_device.type(), DeviceType::CPU);
  EXPECT_EQ(cpu_device.index(), 0);
}

// デバイスの取得テスト
TEST_F(DeviceManagerTest, GetDevice) {
  Device cpu_device = DeviceManager::getDevice(DeviceType::CPU, 0);
  EXPECT_EQ(cpu_device.type(), DeviceType::CPU);
  EXPECT_EQ(cpu_device.index(), 0);
}

// デバイスの可用性チェックテスト
TEST_F(DeviceManagerTest, IsDeviceAvailable) {
  // CPU は常に利用可能
  EXPECT_TRUE(DeviceManager::isDeviceAvailable(DeviceType::CPU, 0));

  // CUDA と Metal はランタイム依存
  // 利用可能かどうかは環境に依存するため、クラッシュしないことを確認
  EXPECT_NO_THROW(DeviceManager::isDeviceAvailable(DeviceType::CUDA, 0));
  EXPECT_NO_THROW(DeviceManager::isDeviceAvailable(DeviceType::METAL, 0));
}

// デバイス数の取得テスト
TEST_F(DeviceManagerTest, GetDeviceCount) {
  // CPU デバイス数は 1
  EXPECT_EQ(DeviceManager::getDeviceCount(DeviceType::CPU), 1);

  // CUDA と Metal はランタイム依存
  EXPECT_GE(DeviceManager::getDeviceCount(DeviceType::CUDA), 0);
  EXPECT_GE(DeviceManager::getDeviceCount(DeviceType::METAL), 0);
}

// Allocator の取得テスト
TEST_F(DeviceManagerTest, GetAllocator) {
  Device cpu_device = cpu();
  auto allocator = DeviceManager::getAllocator(cpu_device);

  EXPECT_NE(allocator, nullptr);
  EXPECT_EQ(allocator->device(), cpu_device);
}

// 同じデバイスに対して同じ Allocator を返すことを確認
TEST_F(DeviceManagerTest, AllocatorCaching) {
  Device cpu_device = cpu();
  auto allocator1 = DeviceManager::getAllocator(cpu_device);
  auto allocator2 = DeviceManager::getAllocator(cpu_device);

  // 同じ shared_ptr を返す（キャッシュされている）
  EXPECT_EQ(allocator1, allocator2);
}

}  // namespace
}  // namespace gradflow
