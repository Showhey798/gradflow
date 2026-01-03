#include "gradflow/autograd/memory_pool.hpp"

#include <algorithm>
#include <new>
#include <stdexcept>
#include <utility>

#include "gradflow/autograd/allocator.hpp"

namespace gradflow {

MemoryPool::MemoryPool(std::shared_ptr<DeviceAllocator> allocator,
                       size_t pool_size)
    : allocator_(std::move(allocator)), pool_size_(pool_size) {
  if (!allocator_) {
    throw std::invalid_argument("Allocator cannot be null");
  }

  if (pool_size_ == 0) {
    throw std::invalid_argument("Pool size must be positive");
  }

  // 初期プールの割り当て
  allocateNewPool(pool_size_);
}

MemoryPool::~MemoryPool() {
  // すべてのプールブロックを解放
  for (const auto& [ptr, size] : pool_blocks_) {
    (void)size;  // size は deallocate では不要だが、ペアとして管理
    allocator_->deallocate(ptr);
  }
}

void* MemoryPool::allocate(size_t size) {
  if (size == 0) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  // アライメント調整
  const size_t alignment = allocator_->alignment();
  const size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

  // Best-fit: 要求サイズ以上の最小ブロックを検索
  auto it = free_blocks_.lower_bound(aligned_size);

  if (it == free_blocks_.end()) {
    // 十分な空きがない → 新しいプールを割り当て
    const size_t new_pool_size = std::max(pool_size_, aligned_size * 2);
    allocateNewPool(new_pool_size);

    // 再度検索
    it = free_blocks_.lower_bound(aligned_size);
    if (it == free_blocks_.end()) {
      throw std::bad_alloc();
    }
  }

  // ブロックを取得
  void* ptr = it->second;
  size_t block_size = it->first;
  free_blocks_.erase(it);
  free_blocks_by_address_.erase(ptr);  // 補助インデックスからも削除

  size_t actual_allocated_size;  // 実際に割り当てられるサイズ

  // ブロックを分割
  if (block_size > aligned_size + kMinBlockSize) {
    // 余剰部分を新しい空きブロックとして登録
    void* remaining_ptr =
        static_cast<char*>(ptr) + static_cast<ptrdiff_t>(aligned_size);
    size_t remaining_size = block_size - aligned_size;
    free_blocks_.insert({remaining_size, remaining_ptr});
    free_blocks_by_address_[remaining_ptr] =
        remaining_size;  // 補助インデックスに追加

    actual_allocated_size = aligned_size;  // 要求サイズ通り
  } else {
    // 分割しない（余剰が小さすぎる）
    actual_allocated_size = block_size;  // ブロック全体を使用
  }

  // 割り当て情報を記録
  allocated_blocks_[ptr] = BlockInfo{ptr, actual_allocated_size};

  // 統計情報の更新
  stats_.total_allocated += actual_allocated_size;
  stats_.current_usage += actual_allocated_size;
  stats_.peak_usage = std::max(stats_.peak_usage, stats_.current_usage);
  stats_.num_allocations++;
  stats_.num_pool_allocations++;

  return ptr;
}

void MemoryPool::deallocate(void* ptr) {
  if (ptr == nullptr) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  // 割り当て情報を取得
  auto alloc_it = allocated_blocks_.find(ptr);
  if (alloc_it == allocated_blocks_.end()) {
    throw std::runtime_error("Invalid pointer: not allocated by this pool");
  }

  BlockInfo info = alloc_it->second;
  allocated_blocks_.erase(alloc_it);

  // 統計情報の更新
  stats_.total_freed += info.size;
  stats_.current_usage -= info.size;
  stats_.num_deallocations++;

  // 隣接する空きブロックとマージ
  void* merged_ptr = info.ptr;
  size_t merged_size = info.size;

  // 前方のブロックとマージ (O(log N))
  auto addr_it = free_blocks_by_address_.lower_bound(merged_ptr);
  if (addr_it != free_blocks_by_address_.begin()) {
    --addr_it;
    void* prev_ptr = addr_it->first;
    size_t prev_size = addr_it->second;
    const void* prev_end =
        static_cast<const char*>(prev_ptr) + static_cast<ptrdiff_t>(prev_size);

    if (prev_end == merged_ptr) {
      // マージ可能
      merged_ptr = prev_ptr;
      merged_size += prev_size;

      // 両方のインデックスから削除
      auto range = free_blocks_.equal_range(prev_size);
      for (auto block_it = range.first; block_it != range.second; ++block_it) {
        if (block_it->second == prev_ptr) {
          free_blocks_.erase(block_it);
          break;
        }
      }
      free_blocks_by_address_.erase(prev_ptr);
    }
  }

  // 後方のブロックとマージ (O(log N))
  void* next_ptr =
      static_cast<char*>(merged_ptr) + static_cast<ptrdiff_t>(merged_size);
  auto next_it = free_blocks_by_address_.find(next_ptr);
  if (next_it != free_blocks_by_address_.end()) {
    size_t next_size = next_it->second;
    merged_size += next_size;

    // 両方のインデックスから削除
    auto range = free_blocks_.equal_range(next_size);
    for (auto block_it = range.first; block_it != range.second; ++block_it) {
      if (block_it->second == next_ptr) {
        free_blocks_.erase(block_it);
        break;
      }
    }
    free_blocks_by_address_.erase(next_ptr);
  }

  // マージされたブロックを登録
  free_blocks_.insert({merged_size, merged_ptr});
  free_blocks_by_address_[merged_ptr] = merged_size;
}

void MemoryPool::reset() {
  std::lock_guard<std::mutex> lock(mutex_);

  // すべての割り当てをクリア
  allocated_blocks_.clear();
  free_blocks_.clear();
  free_blocks_by_address_.clear();

  // プールを再初期化（各ブロックの実際のサイズを使用）
  for (const auto& [ptr, size] : pool_blocks_) {
    free_blocks_.insert({size, ptr});
    free_blocks_by_address_[ptr] = size;
  }

  // 統計情報をリセット（一部は保持）
  stats_.current_usage = 0;
}

MemoryPoolStats MemoryPool::getStatistics() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

void MemoryPool::allocateNewPool(size_t size) {
  void* ptr = allocator_->allocate(size);
  if (ptr == nullptr) {
    throw std::bad_alloc();
  }

  pool_blocks_.push_back({ptr, size});  // サイズも記録
  free_blocks_.insert({size, ptr});
  free_blocks_by_address_[ptr] = size;

  stats_.num_device_allocations++;
}

}  // namespace gradflow
