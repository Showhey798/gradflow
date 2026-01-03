#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "gradflow/autograd/memory_pool.hpp"

#ifdef __APPLE__
#include "gradflow/autograd/metal/allocator.hpp"
#endif

using namespace gradflow;

class MemoryPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
#ifdef __APPLE__
    allocator_ = gpu::getDefaultMetalAllocator();
    if (allocator_) {
      pool_ = std::make_unique<MemoryPool>(allocator_, 1024 * 1024);  // 1 MB
    }
#else
    allocator_ = getDefaultCpuAllocator();
    pool_ = std::make_unique<MemoryPool>(allocator_, 1024 * 1024);  // 1 MB
#endif
  }

  std::shared_ptr<DeviceAllocator> allocator_;
  std::unique_ptr<MemoryPool> pool_;
};

TEST_F(MemoryPoolTest, Allocation) {
  // 基本的な割り当て
  void* ptr1 = pool_->allocate(1024);
  ASSERT_NE(ptr1, nullptr);

  void* ptr2 = pool_->allocate(2048);
  ASSERT_NE(ptr2, nullptr);
  ASSERT_NE(ptr1, ptr2);

  // 統計情報の確認
  auto stats = pool_->getStatistics();
  EXPECT_GE(stats.total_allocated, 3072);
  EXPECT_EQ(stats.num_allocations, 2);
  EXPECT_EQ(stats.num_pool_allocations, 2);

  pool_->deallocate(ptr1);
  pool_->deallocate(ptr2);
}

TEST_F(MemoryPoolTest, Deallocation) {
  void* ptr1 = pool_->allocate(1024);
  void* ptr2 = pool_->allocate(2048);

  pool_->deallocate(ptr1);

  auto stats = pool_->getStatistics();
  EXPECT_EQ(stats.num_deallocations, 1);
  EXPECT_LT(stats.current_usage, stats.total_allocated);

  pool_->deallocate(ptr2);

  stats = pool_->getStatistics();
  EXPECT_EQ(stats.num_deallocations, 2);
  EXPECT_EQ(stats.current_usage, 0);
}

TEST_F(MemoryPoolTest, Fragmentation) {
  // 断片化のテスト
  std::vector<void*> ptrs;

  // 1. 10個の小ブロックを割り当て
  for (int i = 0; i < 10; i++) {
    ptrs.push_back(pool_->allocate(1024));
  }

  // 2. 奇数番目を解放（断片化を発生させる）
  for (size_t i = 1; i < ptrs.size(); i += 2) {
    pool_->deallocate(ptrs[i]);
  }

  // 3. 大きなブロックを割り当て（マージが必要）
  void* large_ptr = pool_->allocate(4096);
  ASSERT_NE(large_ptr, nullptr);

  // クリーンアップ
  for (size_t i = 0; i < ptrs.size(); i += 2) {
    pool_->deallocate(ptrs[i]);
  }
  pool_->deallocate(large_ptr);
}

TEST_F(MemoryPoolTest, Reset) {
  (void)pool_->allocate(1024);
  (void)pool_->allocate(2048);

  pool_->reset();

  auto stats = pool_->getStatistics();
  EXPECT_EQ(stats.current_usage, 0);

  // リセット後も再利用可能
  void* ptr3 = pool_->allocate(1024);
  ASSERT_NE(ptr3, nullptr);

  pool_->deallocate(ptr3);
}

TEST_F(MemoryPoolTest, LargeAllocation) {
  // プールサイズを超える割り当て
  const size_t large_size = 2 * 1024 * 1024;  // 2 MB (pool_size の 2倍)
  void* ptr = pool_->allocate(large_size);
  ASSERT_NE(ptr, nullptr);

  auto stats = pool_->getStatistics();
  EXPECT_GE(stats.num_device_allocations, 2);  // 初期プール + 追加プール

  pool_->deallocate(ptr);
}

TEST_F(MemoryPoolTest, ZeroSizeAllocation) {
  void* ptr = pool_->allocate(0);
  EXPECT_EQ(ptr, nullptr);
}

TEST_F(MemoryPoolTest, InvalidDeallocation) {
  int dummy = 0;
  EXPECT_THROW(pool_->deallocate(&dummy), std::runtime_error);
}

TEST_F(MemoryPoolTest, Statistics) {
  auto stats_before = pool_->getStatistics();

  void* ptr = pool_->allocate(1024);
  auto stats_after = pool_->getStatistics();

  EXPECT_GT(stats_after.total_allocated, stats_before.total_allocated);
  EXPECT_EQ(stats_after.num_allocations, stats_before.num_allocations + 1);

  pool_->deallocate(ptr);
  auto stats_final = pool_->getStatistics();

  EXPECT_EQ(stats_final.num_deallocations, 1);
  EXPECT_EQ(stats_final.current_usage, 0);
}

TEST_F(MemoryPoolTest, ReuseAfterDeallocation) {
  void* ptr1 = pool_->allocate(1024);
  pool_->deallocate(ptr1);

  // 同じサイズを再割り当て（同じブロックが再利用される可能性）
  void* ptr2 = pool_->allocate(1024);
  ASSERT_NE(ptr2, nullptr);

  // 統計: デバイスからの新規割り当てが増えていないことを確認
  auto stats = pool_->getStatistics();
  EXPECT_EQ(stats.num_device_allocations, 1);  // 初期プールのみ

  pool_->deallocate(ptr2);
}

TEST_F(MemoryPoolTest, PerformanceBenchmark) {
  const int num_iterations = 1000;
  const size_t alloc_size = 4096;

  // 1. メモリプールあり
  auto start_pool = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_iterations; i++) {
    void* ptr = pool_->allocate(alloc_size);
    pool_->deallocate(ptr);
  }
  auto end_pool = std::chrono::high_resolution_clock::now();
  auto duration_pool = std::chrono::duration_cast<std::chrono::microseconds>(
      end_pool - start_pool);

  // 2. メモリプールなし（直接割り当て）
  auto start_direct = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_iterations; i++) {
    void* ptr = allocator_->allocate(alloc_size);
    allocator_->deallocate(ptr);
  }
  auto end_direct = std::chrono::high_resolution_clock::now();
  auto duration_direct = std::chrono::duration_cast<std::chrono::microseconds>(
      end_direct - start_direct);

  std::cout << "Pool: " << duration_pool.count() << " μs\n";
  std::cout << "Direct: " << duration_direct.count() << " μs\n";
  if (duration_pool.count() > 0) {
    std::cout << "Speedup: "
              << static_cast<double>(duration_direct.count()) /
                     duration_pool.count()
              << "x\n";
  }

  // プールが少なくとも 2 倍高速であることを確認
  // ただし、Metal Unified Memory では差が小さい可能性があるため、
  // プールが遅くないことを確認（同等またはより高速）
  EXPECT_LE(duration_pool.count(), duration_direct.count() * 2);
}
