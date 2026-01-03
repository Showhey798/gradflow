# Issue #17: Metal での自動微分の詳細設計書

## 1. 調査・リサーチ結果

### Metal での自動微分のアプローチ

Metal GPU 上で自動微分を実装する際、以下の 2 つの主要なアプローチが考えられます：

#### アプローチ 1: Operation クラスを直接 Metal 対応にする
- 既存の `Operation<T>` クラスに Metal 実行パスを追加
- CPU/GPU の切り替えをランタイムで判定
- メリット: 統一されたインターフェース
- デメリット: CPU/GPU の実装が混在し、複雑性が増す

#### アプローチ 2: Metal 専用の Operation クラスを作成
- `MetalAddOperation`, `MetalMatMulOperation` などを別途実装
- デバイスタイプに応じて適切な Operation を選択
- メリット: 実装が明確に分離され、保守性が高い
- デメリット: コードの重複が増える可能性

### 推奨アプローチ: ハイブリッド設計

既存の設計原則と互換性を保ちながら、以下のハイブリッドアプローチを採用します：

1. **Tensor レベルでのデバイス管理**: Tensor がどのデバイス上にあるかを認識
2. **Operation の透過的な実装切り替え**: Operation クラス内で CPU/GPU の実装を自動選択
3. **Metal カーネルの再利用**: 既存の Metal Compute Shader (Issue #15) を活用

### 参考文献

- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) - MSL 公式仕様書
- [Learn performance best practices for Metal shaders](https://developer.apple.com/videos/play/tech-talks/111373/) - Metal パフォーマンスベストプラクティス
- [PyTorch Metal Backend](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/) - PyTorch の Metal 実装参考
- [TensorFlow Metal Plugin](https://developer.apple.com/metal/tensorflow-plugin/) - TensorFlow の Metal 統合事例

---

## 2. 分析と評価

### 現状の課題

#### 完了済みの基盤
- **Phase 3.1 (Issue #14)**: MetalDevice と MetalAllocator が実装済み
- **Phase 3.2 (Issue #15)**: Metal Compute Shader カーネル (add, mul, sub, div, sum, mean, matmul) が実装済み
- **Phase 3.3 (Issue #16)**: MemoryPool による効率的な GPU メモリ管理が実装済み

#### 未実装の部分
- **勾配計算カーネル**: 各 Operation の backward pass を Metal で実行するカーネルが未実装
- **デバイス間の統一インターフェース**: CPU/GPU を透過的に扱う仕組みが不完全
- **Tensor のデバイス管理**: Tensor がどのデバイス上にあるかを追跡する機能が未実装

### 採用すべき設計原則

#### 1. 単一責任の原則 (SRP)
- **MetalGradKernels**: Metal での勾配計算カーネルの管理のみを担当
- **Operation クラス**: CPU/GPU の実装選択ロジックを含むが、各 Operation は単一の数学的演算のみを表現
- **DeviceUtils**: デバイス間のデータ転送ユーティリティ

#### 2. 開放閉鎖原則 (OCP)
- 新しい Operation を追加する際、既存のコードを変更せずに Metal サポートを追加可能
- Metal カーネルと CPU 実装を独立して拡張可能

#### 3. リスコフ置換原則 (LSP)
- CPU 版の Operation と Metal 版の Operation は同じインターフェースを持ち、置換可能
- どちらも `Operation<T>` の契約を満たす

#### 4. 依存性逆転の原則 (DIP)
- 高レベルモジュール (Variable, backward pass) は抽象的な Operation インターフェースに依存
- Metal 固有の実装詳細は低レベルモジュール (.metal, .mm ファイル) に隠蔽

#### 5. RAII (Resource Acquisition Is Initialization)
- Metal リソース (MTLBuffer, MTLCommandBuffer) のライフタイムは C++ オブジェクトのスコープと連動
- デストラクタで自動的に Metal リソースを解放

---

## 3. 推奨アーキテクチャ案

### 設計のコンセプト

**4 層アーキテクチャ**:

```
┌─────────────────────────────────────────────────────────┐
│  Layer 4: High-Level API (Variable, backward)           │
│  - Variable<T>::backward()                              │
│  - 計算グラフのトラバース                                  │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Operation Interface                           │
│  - Operation<T>::forward()                              │
│  - Operation<T>::backward()                             │
│  - CPU/GPU 実装の透過的な切り替え                         │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 2: Device-Specific Implementation                │
│  - CPU: 既存の実装 (ops::add, ops::mul, etc.)            │
│  - Metal: MetalGradKernels                              │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Metal Compute Shader (MSL)                    │
│  - Forward kernels (既存: add_kernel, matmul via MPS)    │
│  - Backward kernels (新規: add_grad, matmul_grad, relu_grad) │
└─────────────────────────────────────────────────────────┘
```

### クラス設計

#### 3.1 MetalGradKernels クラス (新規)

Metal での勾配計算カーネルを管理するクラスです。

```cpp
// include/gradflow/autograd/metal/grad_kernels.hpp
#pragma once

#ifdef __APPLE__

#include <cstddef>
#include <memory>

namespace gradflow {
namespace gpu {

// Forward declaration
class MetalDevice;
class MetalGradKernelsImpl;

/**
 * @brief Metal GPU での勾配計算カーネル管理クラス
 *
 * 各 Operation の backward pass を Metal GPU 上で実行するための
 * カーネルを提供します。
 *
 * 使用例:
 * @code
 *   auto device = MetalDevice::create();
 *   MetalGradKernels grad_kernels(device.get());
 *
 *   // MatMul の勾配計算
 *   grad_kernels.matmul_grad_x(grad_output, y_t, grad_x, m, k, n);
 *   grad_kernels.matmul_grad_y(x_t, grad_output, grad_y, k, m, n);
 * @endcode
 */
class MetalGradKernels {
 public:
  /**
   * @brief MetalGradKernels を構築
   *
   * @param device Metal デバイス (非 null)
   * @throws std::runtime_error カーネルのロード失敗時
   */
  explicit MetalGradKernels(MetalDevice* device);

  ~MetalGradKernels();

  // コピー・ムーブ禁止
  MetalGradKernels(const MetalGradKernels&) = delete;
  MetalGradKernels& operator=(const MetalGradKernels&) = delete;
  MetalGradKernels(MetalGradKernels&&) = delete;
  MetalGradKernels& operator=(MetalGradKernels&&) = delete;

  // ===== Elementwise Operation Gradients =====

  /**
   * @brief Add の勾配計算: grad_x = grad_output, grad_y = grad_output
   *
   * Add の backward は恒等写像なので、単純なコピーで実装可能。
   * 実際には既存の add_kernel を使用せず、直接メモリコピーで実装。
   *
   * @param grad_output 出力の勾配 (GPU メモリ)
   * @param grad_x 入力 x の勾配 (GPU メモリ)
   * @param grad_y 入力 y の勾配 (GPU メモリ)
   * @param size 要素数
   */
  void add_grad(const float* grad_output, float* grad_x, float* grad_y,
                size_t size);

  /**
   * @brief Mul の勾配計算: grad_x = grad_output * y, grad_y = grad_output * x
   *
   * @param grad_output 出力の勾配 (GPU メモリ)
   * @param x 入力 x (GPU メモリ)
   * @param y 入力 y (GPU メモリ)
   * @param grad_x 入力 x の勾配 (GPU メモリ)
   * @param grad_y 入力 y の勾配 (GPU メモリ)
   * @param size 要素数
   */
  void mul_grad(const float* grad_output, const float* x, const float* y,
                float* grad_x, float* grad_y, size_t size);

  // ===== Activation Function Gradients =====

  /**
   * @brief ReLU の勾配計算: grad_x = grad_output * (x > 0 ? 1 : 0)
   *
   * @param grad_output 出力の勾配 (GPU メモリ)
   * @param x 入力 x (GPU メモリ)
   * @param grad_x 入力 x の勾配 (GPU メモリ)
   * @param size 要素数
   */
  void relu_grad(const float* grad_output, const float* x, float* grad_x,
                 size_t size);

  // ===== Matrix Operation Gradients =====

  /**
   * @brief MatMul の x に関する勾配計算: grad_x = grad_output @ y^T
   *
   * @param grad_output 出力の勾配 (GPU メモリ, row-major, size = m * n)
   * @param y_t 転置された y (GPU メモリ, row-major, size = n * k)
   * @param grad_x 入力 x の勾配 (GPU メモリ, row-major, size = m * k)
   * @param m 行列 grad_output の行数
   * @param k 行列 y_t の列数 (= 行列 x の列数)
   * @param n 行列 grad_output の列数
   */
  void matmul_grad_x(const float* grad_output, const float* y_t, float* grad_x,
                     size_t m, size_t k, size_t n);

  /**
   * @brief MatMul の y に関する勾配計算: grad_y = x^T @ grad_output
   *
   * @param x_t 転置された x (GPU メモリ, row-major, size = k * m)
   * @param grad_output 出力の勾配 (GPU メモリ, row-major, size = m * n)
   * @param grad_y 入力 y の勾配 (GPU メモリ, row-major, size = k * n)
   * @param k 行列 grad_y の行数
   * @param m 行列 x_t の列数
   * @param n 行列 grad_output の列数
   */
  void matmul_grad_y(const float* x_t, const float* grad_output, float* grad_y,
                     size_t k, size_t m, size_t n);

  /**
   * @brief GPU 操作を同期
   *
   * 保留中のすべてのコマンドが完了するまで待機します。
   */
  void synchronize();

 private:
  std::unique_ptr<MetalGradKernelsImpl> impl_;
};

}  // namespace gpu
}  // namespace gradflow

#endif  // __APPLE__
```

#### 3.2 Tensor のデバイス認識 (拡張)

現在の Tensor クラスは `DeviceAllocator` を通じてデバイスを管理していますが、
Tensor 自身がどのデバイス上にあるかを認識する必要があります。

```cpp
// include/gradflow/autograd/tensor.hpp (拡張)

enum class DeviceType {
  CPU,
  GPU  // Metal GPU
};

template <typename T>
class Tensor {
 public:
  // 既存のメンバー...

  /**
   * @brief Tensor が存在するデバイスタイプを取得
   * @return デバイスタイプ (CPU または GPU)
   */
  [[nodiscard]] DeviceType deviceType() const {
    if (!storage_) return DeviceType::CPU;
    return storage_->allocator() ? storage_->allocator()->deviceType()
                                 : DeviceType::CPU;
  }

  /**
   * @brief Tensor が GPU 上にあるかチェック
   * @return GPU 上にある場合 true
   */
  [[nodiscard]] bool isOnGPU() const {
    return deviceType() == DeviceType::GPU;
  }

  // 既存のメンバー...
};
```

#### 3.3 Operation クラスの拡張方針

既存の Operation クラス (AddOperation, MulOperation, MatMulOperation, ReLUOperation) を
拡張して、Tensor のデバイスタイプに応じて CPU または Metal 実装を選択します。

**設計パターン: Strategy Pattern + Factory Pattern**

```cpp
// include/gradflow/autograd/ops/add.hpp (拡張版)

template <typename T>
class AddOperation : public Operation<T> {
 public:
  Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
    if (inputs.size() != 2) {
      throw std::invalid_argument("AddOperation requires exactly 2 inputs");
    }

    const auto& x = inputs[0];
    const auto& y = inputs[1];

    // デバイスタイプに応じて実装を選択
    if (x.isOnGPU() && y.isOnGPU()) {
      return forwardMetal(x, y);
    } else {
      return forwardCPU(x, y);
    }
  }

  std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
    // Retrieve input shapes from saved tensors
    auto x_shape = this->getSavedTensor("x_shape_holder").shape();
    auto y_shape = this->getSavedTensor("y_shape_holder").shape();

    if (grad_output.isOnGPU()) {
      return backwardMetal(grad_output, x_shape, y_shape);
    } else {
      return backwardCPU(grad_output, x_shape, y_shape);
    }
  }

 private:
  // CPU 実装 (既存)
  Tensor<T> forwardCPU(const Tensor<T>& x, const Tensor<T>& y) {
    this->saveForBackward("x_shape_holder", Tensor<T>(x.shape()));
    this->saveForBackward("y_shape_holder", Tensor<T>(y.shape()));
    return add(x, y);  // 既存の CPU 実装
  }

  std::vector<Tensor<T>> backwardCPU(const Tensor<T>& grad_output,
                                      const Shape& x_shape,
                                      const Shape& y_shape) {
    auto grad_x = ops::sumToShape(grad_output, x_shape);
    auto grad_y = ops::sumToShape(grad_output, y_shape);
    return {grad_x, grad_y};
  }

  // Metal 実装 (新規)
  Tensor<T> forwardMetal(const Tensor<T>& x, const Tensor<T>& y);
  std::vector<Tensor<T>> backwardMetal(const Tensor<T>& grad_output,
                                        const Shape& x_shape,
                                        const Shape& y_shape);
};
```

---

## 4. Metal Shader 設計

### 4.1 勾配計算カーネルの実装

#### 新規実装が必要なカーネル

```metal
// src/autograd/metal/grad_kernels.metal

#include <metal_stdlib>
using namespace metal;

// ===================================================================
// Elementwise Operation Gradients
// ===================================================================

/**
 * @brief Mul の x に関する勾配カーネル: grad_x = grad_output * y
 */
kernel void mul_grad_x_kernel(device const float* grad_output [[buffer(0)]],
                               device const float* y [[buffer(1)]],
                               device float* grad_x [[buffer(2)]],
                               constant uint& size [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
  if (gid < size) {
    grad_x[gid] = grad_output[gid] * y[gid];
  }
}

/**
 * @brief Mul の y に関する勾配カーネル: grad_y = grad_output * x
 */
kernel void mul_grad_y_kernel(device const float* grad_output [[buffer(0)]],
                               device const float* x [[buffer(1)]],
                               device float* grad_y [[buffer(2)]],
                               constant uint& size [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
  if (gid < size) {
    grad_y[gid] = grad_output[gid] * x[gid];
  }
}

// ===================================================================
// Activation Function Gradients
// ===================================================================

/**
 * @brief ReLU の勾配カーネル: grad_x = grad_output * (x > 0 ? 1 : 0)
 */
kernel void relu_grad_kernel(device const float* grad_output [[buffer(0)]],
                              device const float* x [[buffer(1)]],
                              device float* grad_x [[buffer(2)]],
                              constant uint& size [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
  if (gid < size) {
    float mask = (x[gid] > 0.0f) ? 1.0f : 0.0f;
    grad_x[gid] = grad_output[gid] * mask;
  }
}
```

#### 既存カーネルの再利用

- **MatMul の勾配計算**: 既存の `matmul` カーネル (MPS) を再利用
  - `grad_x = grad_output @ y^T` → matmul(grad_output, y_t)
  - `grad_y = x^T @ grad_output` → matmul(x_t, grad_output)

- **Add の勾配計算**: 勾配は恒等写像なので、Metal の memcpy または既存の add_kernel を使用しない（CPU 側でコピー）

### 4.2 数値安定性の考慮

#### ゼロ除算対策
- Div の backward では、分母が 0 に近い場合に epsilon でクリッピング
- 既存の `div_kernel` と同様のアプローチ

#### オーバーフロー/アンダーフロー対策
- Log の backward: 入力が 0 に近い場合に epsilon を追加
- Exp の backward: 出力が非常に大きい場合にクリッピング

#### 勾配の数値範囲
- すべての勾配計算で float の範囲内に収まるよう、適切なスケーリングを適用

---

## 5. 実装計画

### 5.1 ファイル構成

```
gradflow/
├── include/gradflow/autograd/metal/
│   ├── device.hpp              (既存)
│   ├── allocator.hpp           (既存)
│   ├── kernels.hpp             (既存)
│   └── grad_kernels.hpp        (新規) - 勾配計算カーネルの C++ インターフェース
│
├── src/autograd/metal/
│   ├── device.mm               (既存)
│   ├── allocator.mm            (既存)
│   ├── kernels.mm              (既存)
│   ├── kernels.metal           (既存)
│   ├── grad_kernels.mm         (新規) - 勾配計算カーネルの C++ ラッパー
│   └── grad_kernels.metal      (新規) - 勾配計算カーネルの MSL 実装
│
├── include/gradflow/autograd/ops/
│   ├── add.hpp                 (拡張) - Metal backward 追加
│   ├── mul.hpp                 (拡張) - Metal backward 追加
│   ├── matmul_op.hpp           (拡張) - Metal backward 追加
│   └── relu.hpp                (拡張) - Metal backward 追加
│
└── tests/
    └── test_metal_ops_grad.cpp (新規) - Metal 自動微分のテスト
```

### 5.2 実装ステップ

#### Phase 1: 基盤実装 (1-2 日)

**Step 1.1: MetalGradKernels インターフェース実装**
- [ ] `include/gradflow/autograd/metal/grad_kernels.hpp` 作成
- [ ] `MetalGradKernels` クラスのスケルトン実装

**Step 1.2: Metal Shader カーネル実装**
- [ ] `src/autograd/metal/grad_kernels.metal` 作成
- [ ] `mul_grad_x_kernel`, `mul_grad_y_kernel` 実装
- [ ] `relu_grad_kernel` 実装

**Step 1.3: C++ ラッパー実装**
- [ ] `src/autograd/metal/grad_kernels.mm` 作成
- [ ] Objective-C++ による Metal カーネルの呼び出し実装
- [ ] MTLComputePipelineState の管理実装

#### Phase 2: Operation クラスの拡張 (1-2 日)

**Step 2.1: Tensor のデバイス認識機能追加**
- [ ] `DeviceType` enum の追加
- [ ] `Tensor::deviceType()`, `Tensor::isOnGPU()` メソッド実装
- [ ] `DeviceAllocator::deviceType()` 仮想関数追加

**Step 2.2: AddOperation の Metal 対応**
- [ ] `forwardMetal()` 実装 (既存の MetalKernels::add を使用)
- [ ] `backwardMetal()` 実装 (勾配は恒等写像)

**Step 2.3: MulOperation の Metal 対応**
- [ ] `forwardMetal()` 実装 (既存の MetalKernels::mul を使用)
- [ ] `backwardMetal()` 実装 (MetalGradKernels::mul_grad を使用)

**Step 2.4: ReLUOperation の Metal 対応**
- [ ] `forwardMetal()` 実装 (Metal カーネルで ReLU 実装)
- [ ] `backwardMetal()` 実装 (MetalGradKernels::relu_grad を使用)

**Step 2.5: MatMulOperation の Metal 対応**
- [ ] `forwardMetal()` 実装 (既存の MetalKernels::matmul を使用)
- [ ] `backwardMetal()` 実装 (matmul_grad_x, matmul_grad_y を使用)

#### Phase 3: テスト実装 (1 日)

**Step 3.1: 基本的な勾配テスト**
- [ ] `test_metal_ops_grad.cpp` 作成
- [ ] `MetalOpsGradTest::AddGradient` - Add の GPU 勾配計算テスト
- [ ] `MetalOpsGradTest::MulGradient` - Mul の GPU 勾配計算テスト

**Step 3.2: 複雑な勾配テスト**
- [ ] `MetalOpsGradTest::ReLUGradient` - ReLU の GPU 勾配計算テスト
- [ ] `MetalOpsGradTest::MatMulGradient` - MatMul の GPU 勾配計算テスト

**Step 3.3: CPU/GPU 一致性テスト**
- [ ] CPU 版と GPU 版の勾配が一致することを確認
- [ ] 数値勾配チェック (checkNumericalGradient) を GPU でも実行

#### Phase 4: 最適化とドキュメント (0.5 日)

**Step 4.1: パフォーマンス検証**
- [ ] GPU 版が CPU 版より高速であることを確認
- [ ] 小さいテンソルでのオーバーヘッドを測定

**Step 4.2: ドキュメント更新**
- [ ] PROGRESS.md 更新
- [ ] API ドキュメントの追加

---

## 6. テスト設計

### 6.1 テストケース一覧

#### Test Suite: MetalOpsGradTest

| テストケース | テスト内容 | 期待結果 |
|------------|----------|---------|
| `AddGradient` | Add の GPU 勾配計算 | CPU と GPU の勾配が一致 |
| `MulGradient` | Mul の GPU 勾配計算 | CPU と GPU の勾配が一致 |
| `ReLUGradient` | ReLU の GPU 勾配計算 | CPU と GPU の勾配が一致 |
| `MatMulGradient` | MatMul の GPU 勾配計算 | CPU と GPU の勾配が一致 |
| `AddNumericalGradient` | Add の数値勾配チェック (GPU) | 数値勾配と解析的勾配が一致 |
| `MulNumericalGradient` | Mul の数値勾配チェック (GPU) | 数値勾配と解析的勾配が一致 |
| `ReLUNumericalGradient` | ReLU の数値勾配チェック (GPU) | 数値勾配と解析的勾配が一致 |
| `MatMulNumericalGradient` | MatMul の数値勾配チェック (GPU) | 数値勾配と解析的勾配が一致 |

### 6.2 テスト実装例

```cpp
// tests/test_metal_ops_grad.cpp

TEST_F(MetalOpsGradTest, MatMulGradient) {
  if (!MetalDevice::isAvailable()) {
    GTEST_SKIP() << "Metal is not available";
  }

  // GPU アロケータを作成
  auto device = MetalDevice::create();
  auto allocator = std::make_shared<MetalAllocator>(device.get());

  // GPU 上にテンソルを作成
  Tensor<float> x(Shape({2, 3}), allocator);
  Tensor<float> y(Shape({3, 2}), allocator);

  // データを初期化 (Unified Memory なので CPU から直接書き込み可能)
  for (size_t i = 0; i < x.size(); ++i) {
    x.data()[i] = static_cast<float>(i + 1);
  }
  for (size_t i = 0; i < y.size(); ++i) {
    y.data()[i] = static_cast<float>(i + 1);
  }

  // Forward pass (GPU)
  auto op = std::make_shared<MatMulOperation<float>>();
  auto result = op->forward({x, y});

  // Backward pass (GPU)
  Tensor<float> grad_output(result.shape(), allocator);
  for (size_t i = 0; i < grad_output.size(); ++i) {
    grad_output.data()[i] = 1.0F;
  }

  auto grads = op->backward(grad_output);

  // CPU 版と比較
  Tensor<float> x_cpu(x.shape());
  Tensor<float> y_cpu(y.shape());
  // データをコピー...

  auto op_cpu = std::make_shared<MatMulOperation<float>>();
  auto result_cpu = op_cpu->forward({x_cpu, y_cpu});
  auto grads_cpu = op_cpu->backward(grad_output_cpu);

  // 勾配が一致することを確認
  EXPECT_TRUE(tensorsApproxEqual(grads[0], grads_cpu[0], 1e-5F));
  EXPECT_TRUE(tensorsApproxEqual(grads[1], grads_cpu[1], 1e-5F));
}

TEST_F(MetalOpsGradTest, ReLUGradient) {
  if (!MetalDevice::isAvailable()) {
    GTEST_SKIP() << "Metal is not available";
  }

  auto device = MetalDevice::create();
  auto allocator = std::make_shared<MetalAllocator>(device.get());

  // GPU 上にテンソルを作成
  Tensor<float> x(Shape({1024}), allocator);

  // データを初期化 (正負の値を含む)
  for (size_t i = 0; i < x.size(); ++i) {
    x.data()[i] = static_cast<float>(i) - 512.0F;
  }

  // Forward pass (GPU)
  auto op = std::make_shared<ReLUOperation<float>>();
  auto result = op->forward({x});

  // Backward pass (GPU)
  Tensor<float> grad_output(result.shape(), allocator);
  for (size_t i = 0; i < grad_output.size(); ++i) {
    grad_output.data()[i] = 1.0F;
  }

  auto grads = op->backward(grad_output);

  // CPU 版と比較
  Tensor<float> x_cpu(x.shape());
  for (size_t i = 0; i < x.size(); ++i) {
    x_cpu.data()[i] = x.data()[i];
  }

  auto op_cpu = std::make_shared<ReLUOperation<float>>();
  auto result_cpu = op_cpu->forward({x_cpu});

  Tensor<float> grad_output_cpu(result_cpu.shape());
  for (size_t i = 0; i < grad_output_cpu.size(); ++i) {
    grad_output_cpu.data()[i] = 1.0F;
  }

  auto grads_cpu = op_cpu->backward(grad_output_cpu);

  // 勾配が一致することを確認
  EXPECT_TRUE(tensorsApproxEqual(grads[0], grads_cpu[0], 1e-5F));
}
```

---

## 7. 完了基準

### 7.1 機能完了基準

- [ ] MetalGradKernels クラスが実装され、mul_grad, relu_grad, matmul_grad が動作する
- [ ] AddOperation, MulOperation, ReLUOperation, MatMulOperation が Metal backward をサポート
- [ ] Tensor::deviceType() メソッドが正しくデバイスを認識する
- [ ] Operation クラスが CPU/GPU を透過的に切り替える

### 7.2 品質完了基準

- [ ] すべてのテストがパスする (MetalOpsGradTest::*)
- [ ] CPU と GPU の勾配計算結果が一致する (相対誤差 < 1e-5)
- [ ] 数値勾配チェックがすべてパスする
- [ ] GPU 版が CPU 版より高速 (大きなテンソルで)

### 7.3 ドキュメント完了基準

- [ ] 各クラスに Doxygen コメントが付与されている
- [ ] PROGRESS.md が更新されている
- [ ] 使用例が README に追加されている

---

## 8. 注意事項とリスク

### 8.1 技術的リスク

**リスク 1: Unified Memory の一貫性**
- **問題**: CPU と GPU が同時にメモリにアクセスする場合、データ一貫性の問題が発生する可能性
- **対策**: Metal の synchronize() を適切に呼び出し、GPU 処理完了後に CPU からアクセス

**リスク 2: 小さいテンソルでのオーバーヘッド**
- **問題**: 小さいテンソルでは GPU 起動のオーバーヘッドが演算時間を上回る可能性
- **対策**: テンソルサイズに応じて CPU/GPU を自動選択するヒューリスティックを実装

**リスク 3: ブロードキャストの複雑性**
- **問題**: Metal でのブロードキャストサポートは CPU より複雑
- **対策**: Phase 1 ではブロードキャストなしの単純なケースから実装し、段階的に拡張

### 8.2 保守性リスク

**リスク 4: CPU/GPU コードの重複**
- **問題**: forwardCPU/forwardMetal, backwardCPU/backwardMetal の実装が重複
- **対策**: 共通ロジックをヘルパー関数に抽出し、DRY 原則を維持

**リスク 5: テストの複雑性**
- **問題**: CPU/GPU の両方をテストする必要があり、テストコードが増加
- **対策**: パラメータ化テスト (INSTANTIATE_TEST_SUITE_P) を使用してテストを共通化

---

## 9. パフォーマンス目標

### 9.1 ベンチマーク条件

- **環境**: Apple M1 Max (32 GPU cores)
- **テンソルサイズ**: 1024 x 1024 (float32)
- **演算**: MatMul の forward + backward

### 9.2 期待パフォーマンス

| 演算 | CPU 時間 (ms) | GPU 時間 (ms) | 高速化率 |
|-----|--------------|--------------|---------|
| MatMul forward | 50 | 5 | 10x |
| MatMul backward | 100 | 10 | 10x |
| ReLU forward | 2 | 0.5 | 4x |
| ReLU backward | 2 | 0.5 | 4x |

**注**: 小さいテンソル (< 256 要素) では CPU の方が高速な場合があります。

---

## 10. 今後の拡張

Issue #17 完了後の拡張可能性：

1. **追加の Operation サポート**
   - Sub, Div, Exp, Log, Sigmoid, Tanh の Metal backward 実装
   - Softmax, LayerNorm の Metal backward 実装

2. **Broadcasting の完全サポート**
   - Metal での任意次元 broadcasting 実装
   - GPU メモリ効率を考慮した broadcasting 戦略

3. **自動デバイス選択**
   - テンソルサイズに応じて CPU/GPU を自動選択
   - プロファイリングベースの最適化

4. **Mixed Precision Training**
   - float16 (half) のサポート
   - 動的な精度切り替え

5. **Multi-GPU サポート**
   - 複数 GPU 間でのテンソル分散
   - データ並列学習のサポート

---

## 11. まとめ

Issue #17 の実装により、GradFlow は Metal GPU 上で自動微分を実行できるようになります。
既存の CPU 実装と互換性を保ちながら、Apple Silicon の GPU 能力を最大限に活用します。

**主要な設計判断**:
1. **ハイブリッドアプローチ**: Operation クラス内で CPU/GPU を透過的に切り替え
2. **既存カーネルの再利用**: Issue #15 で実装した Metal カーネルを最大限活用
3. **段階的な実装**: Add, Mul, ReLU, MatMul の順に実装し、段階的に拡張

**期待される効果**:
- 大規模テンソル演算の高速化 (10x 以上)
- Apple Silicon の統合メモリを活用した効率的なデータ転送
- CPU と GPU の統一されたインターフェース

**実装期間**: 3-4 日 (Phase 1-4 の合計)
