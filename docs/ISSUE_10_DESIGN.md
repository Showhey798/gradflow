# Issue #10: 活性化関数の実装 - 技術設計書

## 1. 概要

Phase 2.4 として、ニューラルネットワークで広く使用される活性化関数を実装します。既存の Operation 基底クラスと Issue #9 で実装した基本演算を基盤として、7 つの活性化関数を実装します。

**実装対象**:
- ReLU (Rectified Linear Unit)
- Sigmoid
- Tanh (Hyperbolic Tangent)
- GELU (Gaussian Error Linear Unit)
- LeakyReLU
- Softmax
- LogSoftmax

**依存タスク**:
- ✅ Issue #7: Operation 基底クラス (PR #57)
- ✅ Issue #8: Variable クラス (PR #59)
- ✅ Issue #9: 基本演算の Operation 実装 (PR #60)

## 2. リサーチ結果

### 2.1 PyTorch の実装

PyTorch は `torch.nn.modules.activation` で各活性化関数を提供しています：

- **ReLU**: 非飽和活性化関数。正の値はそのまま、負の値は 0 を出力
- **Sigmoid**: 出力範囲 [0, 1] の非線形関数
- **Tanh**: 出力範囲 [-1, 1] の非線形関数、Sigmoid より勾配が大きい
- **GELU**: Transformer アーキテクチャで推奨される滑らかな活性化関数
- **LeakyReLU**: ReLU の改良版で、負の値にも小さな勾配を持つ

参考:
- [PyTorch Activation Functions](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py)
- [From ReLU to GELU: The PyTorch Activation Handbook](https://medium.com/@rajamails19/from-relu-to-gelu-the-pytorch-activation-handbook-5579b682a3c9)

### 2.2 JAX の実装

JAX は `jax.nn` で活性化関数を提供し、approximate パラメータで高速近似版を選択可能です：

- **GELU**: `jax.nn.gelu(x, approximate=True)` で tanh 近似版を使用可能

参考:
- [JAX GELU Documentation](https://docs.jax.dev/en/latest/_autosummary/jax.nn.gelu.html)

### 2.3 数値安定性: Softmax と LogSumExp トリック

Softmax は指数関数を含むため、数値的にオーバーフロー/アンダーフローが発生しやすい問題があります。これを解決するために **log-sum-exp トリック** を使用します。

**標準的な Softmax**:
```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

**数値安定版 Softmax** (最大値でシフト):
```
max_x = max(x)
softmax(x_i) = exp(x_i - max_x) / Σ_j exp(x_j - max_x)
```

このシフトにより、指数関数に渡される最大値が 0 になり、オーバーフローが防止されます。

**LogSoftmax** も同様に安定化:
```
log_softmax(x_i) = x_i - max_x - log(Σ_j exp(x_j - max_x))
```

参考:
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)
- [Numerically Stable Softmax and Cross Entropy](https://jaykmody.com/blog/stable-softmax/)
- [Accurately computing the log-sum-exp and softmax functions](https://academic.oup.com/imajna/article/41/4/2311/5893596)

## 3. 技術設計

### 3.1 クラス設計

すべての活性化関数は `Operation<T>` を継承し、以下の構造を持ちます：

```cpp
template <typename T>
class ActivationOperation : public Operation<T> {
public:
    // Forward pass: 活性化関数を適用
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override;

    // Backward pass: 勾配を計算
    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override;

    // 操作名（デバッグ用）
    [[nodiscard]] std::string name() const override;
};
```

**設計原則**:
1. **単一責任**: 各クラスは 1 つの活性化関数のみを実装
2. **効率性**: forward で計算した中間値を backward で再利用
3. **数値安定性**: Softmax/LogSoftmax では log-sum-exp トリックを使用
4. **テンプレート**: float/double の両方をサポート

### 3.2 各活性化関数の詳細設計

#### 3.2.1 ReLU (Rectified Linear Unit)

**数式**:
```
forward:  y = max(0, x)
backward: ∂L/∂x = ∂L/∂y * (x > 0 ? 1 : 0)
```

**実装方針**:
- Tensor レベルの `max` 演算を使用
- backward では入力を保存して mask を作成

**保存データ**: `"input"` (backward で mask 作成に使用)

#### 3.2.2 Sigmoid

**数式**:
```
forward:  y = 1 / (1 + exp(-x))
backward: ∂L/∂x = ∂L/∂y * y * (1 - y)
```

**実装方針**:
- Tensor レベルの `exp`, `div`, `add`, `sub` を組み合わせ
- backward では出力 `y` を保存して再利用

**保存データ**: `"output"` (backward で勾配計算に使用)

#### 3.2.3 Tanh

**数式**:
```
forward:  y = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
backward: ∂L/∂x = ∂L/∂y * (1 - y²)
```

**実装方針**:
- Tensor レベルの `tanh` 関数を使用（既存実装を確認）
- または `exp` を使って実装
- backward では出力 `y` を保存

**保存データ**: `"output"` (backward で勾配計算に使用)

#### 3.2.4 GELU (Gaussian Error Linear Unit)

**数式（近似版）**:
```
forward:  y = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
backward: ∂L/∂x = ∂L/∂y * ∂y/∂x
```

ここで、∂y/∂x は複雑ですが、以下のように計算できます：
```
cdf = 0.5 * (1 + tanh(...))
pdf_approximation = (1 - tanh(...)²) * √(2/π) * (1 + 3 * 0.044715 * x²)
∂y/∂x = cdf + 0.5 * x * pdf_approximation
```

**実装方針**:
- tanh 近似版を実装（計算効率が良い）
- backward では入力 `x` と中間値 `tanh_value` を保存

**保存データ**: `"input"`, `"tanh_value"` (backward で勾配計算に使用)

#### 3.2.5 LeakyReLU

**数式**:
```
forward:  y = x > 0 ? x : alpha * x
backward: ∂L/∂x = ∂L/∂y * (x > 0 ? 1 : alpha)
```

**実装方針**:
- `alpha` はコンストラクタで指定（デフォルト: 0.01）
- backward では入力を保存して mask を作成

**保存データ**: `"input"` (backward で mask 作成に使用)

#### 3.2.6 Softmax

**数式（数値安定版）**:
```
forward:
  max_x = max(x, axis=dim)
  exp_shifted = exp(x - max_x)
  y = exp_shifted / sum(exp_shifted, axis=dim)

backward:
  ∂L/∂x = y * (∂L/∂y - Σ(∂L/∂y * y))
```

**実装方針**:
- log-sum-exp トリックで数値安定性を確保
- `dim` パラメータで softmax を適用する次元を指定（デフォルト: -1）
- backward では出力 `y` を保存

**保存データ**: `"output"` (backward で勾配計算に使用)

**数値安定性テスト**:
- 大きな値（1000.0）でオーバーフローしないことを確認
- 小さな値（-1000.0）でアンダーフローしないことを確認

#### 3.2.7 LogSoftmax

**数式（数値安定版）**:
```
forward:
  max_x = max(x, axis=dim)
  log_sum_exp = max_x + log(sum(exp(x - max_x), axis=dim))
  y = x - log_sum_exp

backward:
  ∂L/∂x = ∂L/∂y - sum(∂L/∂y, axis=dim) * exp(y)
```

**実装方針**:
- log-sum-exp トリックで数値安定性を確保
- `dim` パラメータで softmax を適用する次元を指定（デフォルト: -1）
- backward では出力 `y` を保存

**保存データ**: `"output"` (backward で勾配計算に使用)

### 3.3 インターフェース設計

#### 3.3.1 パラメータ化された活性化関数

**LeakyReLU**:
```cpp
template <typename T>
class LeakyReLUOperation : public Operation<T> {
public:
    explicit LeakyReLUOperation(T alpha = static_cast<T>(0.01)) : alpha_(alpha) {}
    // ... forward/backward ...
private:
    T alpha_;
};
```

**Softmax / LogSoftmax**:
```cpp
template <typename T>
class SoftmaxOperation : public Operation<T> {
public:
    explicit SoftmaxOperation(int dim = -1) : dim_(dim) {}
    // ... forward/backward ...
private:
    int dim_;
};
```

#### 3.3.2 ヘルパー関数（必要に応じて）

Softmax/LogSoftmax の実装で共通する処理はヘルパー関数として `op_utils.hpp` に追加します：

```cpp
namespace gradflow::ops {

/**
 * @brief Log-sum-exp トリックで数値安定な softmax を計算
 */
template <typename T>
Tensor<T> stableSoftmax(const Tensor<T>& x, int dim = -1);

/**
 * @brief Log-sum-exp トリックで数値安定な log-softmax を計算
 */
template <typename T>
Tensor<T> stableLogSoftmax(const Tensor<T>& x, int dim = -1);

}  // namespace gradflow::ops
```

### 3.4 必要な Tensor レベルの演算

既存の演算を確認し、不足している演算があれば Phase 1 の範囲で追加が必要です：

**必要な演算**:
- ✅ `add`, `sub`, `mul`, `div` (既存)
- ✅ `exp`, `log`, `sqrt`, `pow` (既存)
- ⏳ `tanh` (確認が必要)
- ⏳ `max` (scalar との比較、またはテンソル全体の最大値)
- ⏳ `sum` (指定した軸での総和) - reduction.hpp で実装済み確認
- ⏳ `reshape` / `unsqueeze` (次元追加)

**確認事項**:
1. `tanh` が Tensor レベルで実装されているか
2. `max(tensor, scalar)` が実装されているか
3. `sum(tensor, axis)` が実装されているか
4. 次元操作（unsqueeze, squeeze）が実装されているか

## 4. テスト戦略

### 4.1 テスト構成

各活性化関数に対して以下のテストを実施します：

```cpp
// tests/test_activation_ops.cpp
class ActivationOpsTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // ヘルパー関数
    bool approx_equal(float a, float b, float epsilon = 1e-5F);
};

// 各活性化関数のテストケース
// 1. Forward テスト
// 2. Backward テスト
// 3. 数値勾配チェック
// 4. エッジケース
```

### 4.2 テストケース詳細

#### 4.2.1 Forward テスト

各活性化関数の出力が期待値と一致するかを確認します。

**ReLU**:
```cpp
TEST_F(ActivationOpsTest, ReLUForward) {
    auto x = Tensor<float>({-2.0F, -1.0F, 0.0F, 1.0F, 2.0F});
    auto op = std::make_shared<ReLUOperation<float>>();
    auto result = op->forward({x});

    EXPECT_FLOAT_EQ(result[{0}], 0.0F);
    EXPECT_FLOAT_EQ(result[{1}], 0.0F);
    EXPECT_FLOAT_EQ(result[{2}], 0.0F);
    EXPECT_FLOAT_EQ(result[{3}], 1.0F);
    EXPECT_FLOAT_EQ(result[{4}], 2.0F);
}
```

#### 4.2.2 Backward テスト

勾配が正しく伝播するかを確認します。

**Sigmoid**:
```cpp
TEST_F(ActivationOpsTest, SigmoidBackward) {
    auto x = Tensor<float>({0.0F});
    auto op = std::make_shared<SigmoidOperation<float>>();
    auto output = op->forward({x});

    // sigmoid(0) = 0.5
    EXPECT_FLOAT_EQ(output[{0}], 0.5F);

    auto grad_output = Tensor<float>({1.0F});
    auto grads = op->backward(grad_output);

    // grad = y * (1 - y) = 0.5 * 0.5 = 0.25
    EXPECT_FLOAT_EQ(grads[0][{0}], 0.25F);
}
```

#### 4.2.3 数値勾配チェック

既存の `checkNumericalGradient` 関数を使用して、自動微分の勾配が数値勾配と一致するかを確認します。

```cpp
TEST_F(ActivationOpsTest, TanhNumericalGradient) {
    auto x = Tensor<float>({-1.0F, 0.0F, 1.0F});
    auto op = std::make_shared<TanhOperation<float>>();

    bool grad_correct = ops::test::checkNumericalGradient(
        *op, {x}, {}, 1e-4F, 1e-2F);
    EXPECT_TRUE(grad_correct);
}
```

#### 4.2.4 Softmax 数値安定性テスト

Softmax の log-sum-exp トリックが正しく動作するかを確認します。

```cpp
TEST_F(ActivationOpsTest, SoftmaxNumericalStability) {
    // 大きな値でオーバーフローしないか
    auto x_large = Tensor<float>({1000.0F, 1000.0F, 1000.0F});
    auto op_large = std::make_shared<SoftmaxOperation<float>>();
    auto result_large = op_large->forward({x_large});

    // 各要素が有限値であることを確認
    EXPECT_TRUE(std::isfinite(result_large[{0}]));
    EXPECT_TRUE(std::isfinite(result_large[{1}]));
    EXPECT_TRUE(std::isfinite(result_large[{2}]));

    // 総和が 1 であることを確認
    float sum = result_large[{0}] + result_large[{1}] + result_large[{2}];
    EXPECT_NEAR(sum, 1.0F, 1e-5F);

    // 小さな値でアンダーフローしないか
    auto x_small = Tensor<float>({-1000.0F, -1000.0F, -1000.0F});
    auto op_small = std::make_shared<SoftmaxOperation<float>>();
    auto result_small = op_small->forward({x_small});

    EXPECT_TRUE(std::isfinite(result_small[{0}]));
    EXPECT_TRUE(std::isfinite(result_small[{1}]));
    EXPECT_TRUE(std::isfinite(result_small[{2}]));
}
```

#### 4.2.5 LeakyReLU パラメータテスト

alpha パラメータが正しく動作するかを確認します。

```cpp
TEST_F(ActivationOpsTest, LeakyReLUParameter) {
    float alpha = 0.2F;
    auto x = Tensor<float>({-2.0F, -1.0F, 0.0F, 1.0F, 2.0F});
    auto op = std::make_shared<LeakyReLUOperation<float>>(alpha);
    auto result = op->forward({x});

    EXPECT_FLOAT_EQ(result[{0}], -2.0F * alpha);
    EXPECT_FLOAT_EQ(result[{1}], -1.0F * alpha);
    EXPECT_FLOAT_EQ(result[{2}], 0.0F);
    EXPECT_FLOAT_EQ(result[{3}], 1.0F);
    EXPECT_FLOAT_EQ(result[{4}], 2.0F);
}
```

#### 4.2.6 多次元テンソルでのテスト

2D テンソルで各活性化関数が正しく動作するかを確認します。

```cpp
TEST_F(ActivationOpsTest, ReLU2D) {
    auto x = Tensor<float>({{-1.0F, 2.0F}, {3.0F, -4.0F}});
    auto op = std::make_shared<ReLUOperation<float>>();
    auto result = op->forward({x});

    EXPECT_EQ(result.shape(), Shape({2, 2}));
    EXPECT_FLOAT_EQ((result[{0, 0}]), 0.0F);
    EXPECT_FLOAT_EQ((result[{0, 1}]), 2.0F);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 3.0F);
    EXPECT_FLOAT_EQ((result[{1, 1}]), 0.0F);
}
```

### 4.3 テスト実行計画

**Phase 1: 単体テスト**
1. 各活性化関数の forward/backward を個別にテスト
2. 数値勾配チェックで勾配の正確性を確認
3. エッジケース（極端な値、境界値）をテスト

**Phase 2: 統合テスト**
1. 複数の活性化関数を連続して適用
2. Variable クラスとの統合（Issue #8 の成果を利用）

**Phase 3: パフォーマンステスト**
1. 大規模テンソル（1000x1000 など）での性能確認
2. メモリ使用量の確認

### 4.4 期待される結果

**すべてのテストが pass する基準**:
- ✅ Forward テスト: 各活性化関数が正しい出力を生成
- ✅ Backward テスト: 勾配が正しく計算される
- ✅ 数値勾配チェック: 相対誤差 < 1e-2
- ✅ Softmax 数値安定性: 極端な値でもオーバーフロー/アンダーフローしない
- ✅ パラメータテスト: LeakyReLU の alpha が正しく動作
- ✅ 多次元テスト: 2D 以上のテンソルでも動作

## 5. 実装ファイル構成

### 5.1 ヘッダーファイル

各活性化関数を個別のヘッダーファイルで実装します：

```
include/gradflow/autograd/ops/
├── relu.hpp           # ReLU 実装
├── sigmoid.hpp        # Sigmoid 実装
├── tanh.hpp           # Tanh 実装
├── gelu.hpp           # GELU 実装
├── leaky_relu.hpp     # LeakyReLU 実装
├── softmax.hpp        # Softmax 実装
└── log_softmax.hpp    # LogSoftmax 実装
```

**ファイル構造例** (`relu.hpp`):
```cpp
#pragma once

#include "../operation.hpp"
#include "../tensor.hpp"
#include "elementwise.hpp"  // max, mul などのヘルパー関数

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gradflow {

/**
 * @brief ReLU (Rectified Linear Unit) operation
 *
 * Forward:
 *   y = max(0, x)
 *
 * Backward:
 *   ∂L/∂x = ∂L/∂y * (x > 0 ? 1 : 0)
 */
template <typename T>
class ReLUOperation : public Operation<T> {
public:
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override;
    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override;
    [[nodiscard]] std::string name() const override { return "ReLUOperation"; }
};

}  // namespace gradflow
```

### 5.2 テストファイル

```
tests/
├── test_activation_ops.cpp  # 活性化関数の統合テスト
└── test_activation_grad.cpp # 数値勾配チェック
```

### 5.3 ドキュメント

```
docs/
├── ISSUE_10_DESIGN.md       # 本設計書
└── ACTIVATION_FUNCTIONS.md  # 活性化関数の使用ガイド（実装後に作成）
```

## 6. 実装の優先順位

段階的に実装を進めます：

**Phase 1: 基本的な活性化関数** (優先度: 高)
1. ReLU
2. Sigmoid
3. Tanh

**Phase 2: パラメータ付き活性化関数** (優先度: 中)
4. LeakyReLU
5. GELU

**Phase 3: 正規化系活性化関数** (優先度: 中)
6. Softmax (数値安定性に注意)
7. LogSoftmax (数値安定性に注意)

**Phase 4: テストと検証** (優先度: 高)
8. 数値勾配チェック
9. Softmax 数値安定性テスト
10. 統合テスト

## 7. リスクと対策

### 7.1 数値安定性の問題

**リスク**: Softmax/LogSoftmax でオーバーフロー/アンダーフロー

**対策**:
- log-sum-exp トリックを必ず使用
- 極端な値でのテストを追加
- 浮動小数点演算の精度に注意

### 7.2 勾配消失/爆発

**リスク**: Sigmoid/Tanh で勾配が消失する可能性

**対策**:
- 実装は正しく行い、使用者側で適切な初期化を促す
- ドキュメントで注意点を明記

### 7.3 パフォーマンス

**リスク**: GELU の複雑な計算で性能低下

**対策**:
- tanh 近似版を実装（PyTorch/JAX と同様）
- 必要に応じて CPU 最適化

### 7.4 Tensor レベル演算の不足

**リスク**: 必要な演算（tanh, max など）が実装されていない

**対策**:
- Phase 1 の範囲で必要な演算を追加
- または Operation レベルで低レベル実装

## 8. 完了基準

以下の条件をすべて満たすことで完了とします：

- ✅ 7 つの活性化関数がすべて実装されている
- ✅ 各活性化関数の forward と backward が正しく動作する
- ✅ 数値勾配チェックがすべて pass する
- ✅ Softmax/LogSoftmax の数値安定性テストが pass する
- ✅ すべてのテストが pass する
- ✅ すべての CI チェックが pass する (build, test, clang-tidy, sanitizers, format)
- ✅ ドキュメントが整備されている

## 9. 次のステップ

Issue #10 完了後、次の Phase 2.5 へ進みます：

- **Phase 2.5**: 損失関数の実装 (CrossEntropyLoss, MSELoss, BCELoss など)
- **Phase 2.6**: Optimizer の実装 (SGD, Adam など)

## 10. 参考文献

### PyTorch
- [PyTorch Activation Functions](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py)
- [From ReLU to GELU: The PyTorch Activation Handbook](https://medium.com/@rajamails19/from-relu-to-gelu-the-pytorch-activation-handbook-5579b682a3c9)
- [PyTorch GELU Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html)

### JAX
- [JAX GELU Documentation](https://docs.jax.dev/en/latest/_autosummary/jax.nn.gelu.html)

### 数値安定性
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)
- [Numerically Stable Softmax and Cross Entropy](https://jaykmody.com/blog/stable-softmax/)
- [Accurately computing the log-sum-exp and softmax functions](https://academic.oup.com/imajna/article/41/4/2311/5893596)

### その他
- [GELU: Gaussian Error Linear Unit](https://alaaalatif.github.io/2019-04-11-gelu/)
