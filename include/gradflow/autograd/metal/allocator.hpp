#pragma once

#ifdef __APPLE__

#include <cstddef>
#include <memory>
#include <unordered_map>

#include "gradflow/autograd/allocator.hpp"
#include "gradflow/autograd/metal/device.hpp"

namespace gradflow {
namespace gpu {

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
  explicit MetalAllocator(MetalDevice* device,
                          size_t alignment = kDefaultAlignment);

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
  [[nodiscard]] Device device() const override;

  /**
   * @brief アライメント要件を取得
   * @return アライメント (bytes)
   */
  [[nodiscard]] size_t alignment() const override { return alignment_; }

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

  /**
   * @brief メモリポインタから MTLBuffer を取得
   *
   * このメソッドは Metal カーネル実行時に使用します。
   * allocate() で確保されたメモリポインタから、対応する MTLBuffer
   * を取得します。
   *
   * @param ptr allocate() で確保されたメモリポインタ
   * @return MTLBuffer (void* にキャスト)、見つからない場合は nullptr
   *
   * @note 返されるポインタは void* ですが、実際には id<MTLBuffer> です。
   *       使用側で (__bridge id<MTLBuffer>) でキャストしてください。
   */
  void* getBuffer(void* ptr);

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

}  // namespace gpu
}  // namespace gradflow

#endif  // __APPLE__
