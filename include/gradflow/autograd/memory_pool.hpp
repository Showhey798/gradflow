#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gradflow {

// Forward declaration
class DeviceAllocator;

/**
 * @brief メモリプールの統計情報
 */
struct MemoryPoolStats {
  /// 総割り当てサイズ (bytes)
  size_t total_allocated = 0;
  /// 総解放サイズ (bytes)
  size_t total_freed = 0;
  /// 現在の使用量 (bytes)
  size_t current_usage = 0;
  /// ピーク使用量 (bytes)
  size_t peak_usage = 0;
  /// 割り当て回数
  size_t num_allocations = 0;
  /// 解放回数
  size_t num_deallocations = 0;
  /// プールからの割り当て回数
  size_t num_pool_allocations = 0;
  /// デバイスからの直接割り当て回数
  size_t num_device_allocations = 0;
};

/**
 * @brief 効率的なメモリ管理のためのメモリプール
 *
 * MemoryPool は大きなメモリブロックを事前確保し、内部で細かく管理することで
 * 頻繁なメモリ割り当て・解放のオーバーヘッドを削減します。
 *
 * 特徴:
 * - Best-fit アロケーション: 要求サイズに最も近い空きブロックを選択
 * - 断片化緩和: 隣接する空きブロックを自動的にマージ
 * - スレッドセーフ: std::mutex による同期
 * - 統計情報: 詳細なメモリ使用状況を追跡
 *
 * 使用例:
 * @code
 *   auto allocator = getDefaultMetalAllocator();
 *   MemoryPool pool(allocator, 1024 * 1024 * 1024); // 1GB
 *
 *   void* ptr1 = pool.allocate(1024);
 *   void* ptr2 = pool.allocate(2048);
 *
 *   pool.deallocate(ptr1);
 *   pool.deallocate(ptr2);
 *
 *   auto stats = pool.getStatistics();
 *   std::cout << "Peak usage: " << stats.peak_usage << " bytes\n";
 * @endcode
 */
class MemoryPool {
 public:
  /**
   * @brief デフォルトのプールサイズ (256 MB)
   */
  static constexpr size_t kDefaultPoolSize = 256ULL * 1024 * 1024;

  /**
   * @brief 最小ブロックサイズ (256 bytes)
   *
   * これより小さいブロックは分割しません。
   */
  static constexpr size_t kMinBlockSize = 256;

  /**
   * @brief MemoryPool を構築
   *
   * @param allocator 基底となるデバイスアロケータ
   * @param pool_size 初期プールサイズ (bytes, デフォルト: 256 MB)
   * @throws std::invalid_argument allocator が null の場合
   * @throws std::bad_alloc 初期プールの割り当てに失敗した場合
   */
  explicit MemoryPool(std::shared_ptr<DeviceAllocator> allocator,
                      size_t pool_size = kDefaultPoolSize);

  ~MemoryPool();

  // コピー禁止、ムーブ禁止
  MemoryPool(const MemoryPool&) = delete;
  MemoryPool& operator=(const MemoryPool&) = delete;
  MemoryPool(MemoryPool&&) = delete;
  MemoryPool& operator=(MemoryPool&&) = delete;

  /**
   * @brief メモリを割り当て
   *
   * Best-fit アルゴリズムで最適な空きブロックを検索し、
   * 必要に応じて分割します。
   *
   * @param size 要求サイズ (bytes)
   * @return 割り当てられたメモリへのポインタ
   * @throws std::bad_alloc 割り当て失敗時
   */
  void* allocate(size_t size);

  /**
   * @brief メモリを解放
   *
   * 解放されたブロックは隣接する空きブロックと自動的にマージされます。
   *
   * @param ptr 解放するメモリポインタ
   * @throws std::runtime_error 無効なポインタの場合
   */
  void deallocate(void* ptr);

  /**
   * @brief プールをリセット
   *
   * すべての割り当てを解放し、プールを初期状態に戻します。
   *
   * @note 外部で保持されているポインタはすべて無効になります
   */
  void reset();

  /**
   * @brief 統計情報を取得
   *
   * @return MemoryPoolStats 構造体
   */
  [[nodiscard]] MemoryPoolStats getStatistics() const;

 private:
  /**
   * @brief ブロック情報
   */
  struct BlockInfo {
    void* ptr;    ///< ブロックの開始アドレス
    size_t size;  ///< ブロックサイズ (bytes)
  };

  /**
   * @brief 新しいプールブロックを割り当て
   *
   * @param size 要求サイズ (bytes)
   */
  void allocateNewPool(size_t size);

  std::shared_ptr<DeviceAllocator> allocator_;  ///< 基底アロケータ
  size_t pool_size_;                            ///< プールサイズ

  // 空きブロックの管理 (サイズでソート)
  // multimap を使用することで Best-fit が O(log N) で実現可能
  std::multimap<size_t, void*> free_blocks_;

  // アドレスベースの補助インデックス (O(log N) で隣接検索を実現)
  std::map<void*, size_t> free_blocks_by_address_;

  // 割り当て済みブロックの管理
  std::unordered_map<void*, BlockInfo> allocated_blocks_;

  // プールとして確保された大きなメモリブロック (ptr, size)
  std::vector<std::pair<void*, size_t>> pool_blocks_;

  // 統計情報
  MemoryPoolStats stats_;

  // スレッドセーフのための mutex
  mutable std::mutex mutex_;
};

}  // namespace gradflow
