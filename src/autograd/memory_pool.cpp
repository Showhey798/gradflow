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
  for (void* ptr : pool_blocks_) {
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
  size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

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

  // ブロックを分割
  if (block_size > aligned_size + kMinBlockSize) {
    // 余剰部分を新しい空きブロックとして登録
    void* remaining_ptr =
        static_cast<char*>(ptr) + static_cast<ptrdiff_t>(aligned_size);
    size_t remaining_size = block_size - aligned_size;
    free_blocks_.insert({remaining_size, remaining_ptr});
  } else {
    // 分割しない（余剰が小さすぎる）
    aligned_size = block_size;
  }

  // 割り当て情報を記録
  allocated_blocks_[ptr] = BlockInfo{ptr, aligned_size, false};

  // 統計情報の更新
  stats_.total_allocated += aligned_size;
  stats_.current_usage += aligned_size;
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
  auto it = allocated_blocks_.find(ptr);
  if (it == allocated_blocks_.end()) {
    throw std::runtime_error("Invalid pointer: not allocated by this pool");
  }

  BlockInfo info = it->second;
  allocated_blocks_.erase(it);

  // 統計情報の更新
  stats_.total_freed += info.size;
  stats_.current_usage -= info.size;
  stats_.num_deallocations++;

  // 隣接する空きブロックとマージ
  void* merged_ptr = info.ptr;
  size_t merged_size = info.size;

  // 前方のブロックとマージ
  auto prev = findPreviousFreeBlock(merged_ptr);
  if (prev != free_blocks_.end()) {
    merged_ptr = prev->second;
    merged_size += prev->first;
    free_blocks_.erase(prev);
  }

  // 後方のブロックとマージ
  auto next = findNextFreeBlock(merged_ptr, merged_size);
  if (next != free_blocks_.end()) {
    merged_size += next->first;
    free_blocks_.erase(next);
  }

  // マージされたブロックを登録
  free_blocks_.insert({merged_size, merged_ptr});
}

void MemoryPool::reset() {
  std::lock_guard<std::mutex> lock(mutex_);

  // すべての割り当てをクリア
  allocated_blocks_.clear();
  free_blocks_.clear();

  // プールを再初期化
  for (void* ptr : pool_blocks_) {
    free_blocks_.insert({pool_size_, ptr});
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

  pool_blocks_.push_back(ptr);
  free_blocks_.insert({size, ptr});

  stats_.num_device_allocations++;
}

auto MemoryPool::findPreviousFreeBlock(void* ptr)
    -> std::multimap<size_t, void*>::iterator {
  for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
    void* block_end =
        static_cast<char*>(it->second) + static_cast<ptrdiff_t>(it->first);
    if (block_end == ptr) {
      return it;
    }
  }
  return free_blocks_.end();
}

auto MemoryPool::findNextFreeBlock(void* ptr, size_t size)
    -> std::multimap<size_t, void*>::iterator {
  void* block_end = static_cast<char*>(ptr) + static_cast<ptrdiff_t>(size);
  for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
    if (it->second == block_end) {
      return it;
    }
  }
  return free_blocks_.end();
}

}  // namespace gradflow
