#pragma once

#ifdef __APPLE__

#include <cstddef>
#include <memory>
#include <string>

namespace gradflow {
namespace gpu {

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
    [[nodiscard]] std::string name() const;

    /**
     * @brief 推奨される最大ワーキングセットサイズを取得
     *
     * GPU が効率的に処理できる最大メモリサイズ（バイト）。
     * 通常は物理メモリの約 75%。
     *
     * @return 推奨最大メモリサイズ (bytes)
     */
    [[nodiscard]] size_t recommendedMaxWorkingSetSize() const;

    /**
     * @brief Unified Memory をサポートしているかチェック
     * @return Apple Silicon なら常に true
     */
    [[nodiscard]] bool hasUnifiedMemory() const;

    /**
     * @brief 内部実装へのアクセス (Allocator 用)
     * @internal この関数は内部実装でのみ使用されます
     * @return 内部実装ポインタ
     */
    MetalDeviceImpl* impl() const { return impl_.get(); }

    /**
     * @brief Metal デバイスハンドルを取得 (Allocator 用)
     * @internal この関数は内部実装でのみ使用されます
     * @return Metal デバイスハンドル (void* として id<MTLDevice> を返す)
     */
    void* getMetalDevice() const;

    /**
     * @brief Metal コマンドキューハンドルを取得 (Allocator 用)
     * @internal この関数は内部実装でのみ使用されます
     * @return Metal コマンドキューハンドル (void* として id<MTLCommandQueue> を返す)
     */
    void* getMetalCommandQueue() const;

private:
    MetalDevice();  // Private constructor (factory pattern)

    std::unique_ptr<MetalDeviceImpl> impl_;
};

}  // namespace gpu
}  // namespace gradflow

#endif  // __APPLE__
