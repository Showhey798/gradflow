# Issue #9: 基本演算の Operation 実装 - 設計ドキュメント

## 目次
- [概要](#概要)
- [設計原則](#設計原則)
- [アーキテクチャ](#アーキテクチャ)
- [実装詳細](#実装詳細)
- [テスト戦略](#テスト戦略)
- [実装タスクリスト](#実装タスクリスト)
- [完了基準](#完了基準)

---

## 概要

### 目的
自動微分機能を実現するため、基本的な演算（Add, Sub, Mul, Div, MatMul, Pow, Exp, Log, Sqrt）の Operation クラスを実装する。各演算は forward pass（順伝播）と backward pass（逆伝播）を持ち、計算グラフの一部として動作する。

### 依存関係
- ✅ Issue #7: Operation 基底クラス（PR #57: マージ済み）
- ✅ Issue #8: Variable クラス（PR #59: レビュー待ち）
- ✅ 既存の Tensor レベルの演算（elementwise.hpp, matmul.hpp）

### 実装スコープ
9 つの Operation クラス + 包括的なテストスイート

---

## 設計原則

### 1. 単一責任の原則 (Single Responsibility Principle)
各 Operation クラスは、単一の演算に対する forward と backward のみを実装する。

### 2. DRY (Don't Repeat Yourself)
- Tensor レベルの演算（`add`, `mul`, `matmul` など）は既に実装済みなので、それらを再利用する
- Broadcasting 対応の勾配処理は共通のヘルパー関数で実装する
- 数値勾配チェックは汎用的なテンプレート関数で実装する

### 3. 型安全性
- テンプレートを使用して、任意の数値型 (`float`, `double`) に対応
- コンパイル時の型チェックを最大限活用

### 4. メモリ効率
- 不要な Tensor のコピーを避ける（view や参照を活用）
- forward 時に backward で必要な値のみを保存

### 5. 数値安定性
- 0 除算を避ける（DivOperation, LogOperation, SqrtOperation）
- log(0), sqrt(負数) などの不正な入力に対するエラーハンドリング

---

## アーキテクチャ

### ファイル構成

```
include/gradflow/autograd/
├── operation.hpp              # 基底クラス（既存）
├── variable.hpp               # Variable クラス（既存）
└── ops/
    ├── elementwise.hpp        # Tensor レベルの関数（既存）
    ├── matmul.hpp             # Tensor レベルの関数（既存）
    ├── reduction.hpp          # Tensor レベルの関数（既存）
    ├── op_utils.hpp           # NEW: 共通ユーティリティ関数
    ├── add.hpp                # NEW: AddOperation
    ├── sub.hpp                # NEW: SubOperation
    ├── mul.hpp                # NEW: MulOperation
    ├── div.hpp                # NEW: DivOperation
    ├── matmul_op.hpp          # NEW: MatMulOperation (既存の matmul.hpp と区別)
    ├── pow.hpp                # NEW: PowOperation
    ├── exp.hpp                # NEW: ExpOperation
    ├── log.hpp                # NEW: LogOperation
    └── sqrt.hpp               # NEW: SqrtOperation

tests/
├── test_ops.cpp               # Tensor レベルのテスト（既存）
└── test_ops_grad.cpp          # NEW: Operation の勾配テスト
```

### クラス階層

```
Operation<T> (抽象基底クラス)
    ↑
    ├── AddOperation<T>
    ├── SubOperation<T>
    ├── MulOperation<T>
    ├── DivOperation<T>
    ├── MatMulOperation<T>
    ├── PowOperation<T>
    ├── ExpOperation<T>
    ├── LogOperation<T>
    └── SqrtOperation<T>
```

---

## 実装詳細

### Phase 1: 共通ユーティリティ（op_utils.hpp）

Broadcasting に対応した勾配計算のためのヘルパー関数を実装する。

#### 1.1 Broadcasting 勾配の調整関数

```cpp
namespace gradflow {
namespace ops {

/**
 * @brief Adjust gradient for broadcasting
 *
 * When broadcasting happens during forward pass, the gradient needs to be
 * summed over the broadcasted dimensions during backward pass.
 *
 * Example:
 *   Forward: a[3, 1] + b[3, 4] -> c[3, 4]  (b is broadcasted)
 *   Backward: grad_a needs to be summed over dimension 1
 *
 * @tparam T Element type
 * @param grad Gradient tensor (same shape as forward output)
 * @param target_shape Target shape (shape of the input during forward)
 * @return Adjusted gradient tensor (same shape as target_shape)
 */
template <typename T>
Tensor<T> sumToShape(const Tensor<T>& grad, const Shape& target_shape);

}  // namespace ops
}  // namespace gradflow
```

**実装方針**:
1. `grad` の形状と `target_shape` を比較
2. Broadcasting された次元を特定
3. その次元に沿って sum を実行
4. 結果を reshape して `target_shape` に合わせる

#### 1.2 数値勾配チェック関数（テスト用）

```cpp
namespace gradflow {
namespace ops {
namespace test {

/**
 * @brief Numerical gradient checker
 *
 * Compares automatic differentiation gradient with numerical gradient
 * using finite difference method.
 *
 * @tparam T Element type
 * @param op Operation to test
 * @param inputs Input tensors
 * @param output_index Which output element to use for gradient check
 * @param epsilon Finite difference step size (default: 1e-4)
 * @param tolerance Acceptable error (default: 1e-3)
 * @return True if all gradients are within tolerance
 */
template <typename T>
bool checkNumericalGradient(
    Operation<T>& op,
    const std::vector<Tensor<T>>& inputs,
    const std::vector<size_t>& output_index = {},
    T epsilon = static_cast<T>(1e-4),
    T tolerance = static_cast<T>(1e-3)
);

}  // namespace test
}  // namespace ops
}  // namespace gradflow
```

### Phase 2: Element-wise Binary Operations

#### 2.1 AddOperation

**数学的定義**:
- Forward: `z = x + y`
- Backward: `∂L/∂x = ∂L/∂z`, `∂L/∂y = ∂L/∂z`

**実装**:

```cpp
template <typename T>
class AddOperation : public Operation<T> {
public:
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 2) {
            throw std::invalid_argument("AddOperation requires exactly 2 inputs");
        }

        const auto& x = inputs[0];
        const auto& y = inputs[1];

        // Save input shapes for backward
        saveForBackward("x_shape", /* Shape as Tensor */);
        saveForBackward("y_shape", /* Shape as Tensor */);

        return add(x, y);  // Use existing Tensor-level function
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        // For addition, gradient is passed through unchanged
        // But we need to handle broadcasting
        auto x_shape = /* retrieve saved shape */;
        auto y_shape = /* retrieve saved shape */;

        auto grad_x = ops::sumToShape(grad_output, x_shape);
        auto grad_y = ops::sumToShape(grad_output, y_shape);

        return {grad_x, grad_y};
    }

    std::string name() const override { return "AddOperation"; }
};
```

**注意点**:
- Broadcasting が発生した場合、逆伝播時に勾配を適切な形状に調整する必要がある
- Shape を保存する方法として、Tensor として保存するか、別途メンバ変数を用意するか検討が必要

#### 2.2 SubOperation

**数学的定義**:
- Forward: `z = x - y`
- Backward: `∂L/∂x = ∂L/∂z`, `∂L/∂y = -∂L/∂z`

**実装**:
AddOperation と類似だが、`grad_y` には `-1` を掛ける。

#### 2.3 MulOperation

**数学的定義**:
- Forward: `z = x * y`
- Backward: `∂L/∂x = ∂L/∂z * y`, `∂L/∂y = ∂L/∂z * x`

**実装**:
```cpp
template <typename T>
class MulOperation : public Operation<T> {
public:
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        const auto& x = inputs[0];
        const auto& y = inputs[1];

        // Save inputs for backward
        saveForBackward("x", x);
        saveForBackward("y", y);

        return mul(x, y);
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto x = getSavedTensor("x");
        auto y = getSavedTensor("y");

        auto grad_x = mul(grad_output, y);
        auto grad_y = mul(grad_output, x);

        // Handle broadcasting
        grad_x = ops::sumToShape(grad_x, x.shape());
        grad_y = ops::sumToShape(grad_y, y.shape());

        return {grad_x, grad_y};
    }

    std::string name() const override { return "MulOperation"; }
};
```

#### 2.4 DivOperation

**数学的定義**:
- Forward: `z = x / y`
- Backward:
  - `∂L/∂x = ∂L/∂z / y`
  - `∂L/∂y = -∂L/∂z * x / (y * y)`

**実装**:
```cpp
template <typename T>
class DivOperation : public Operation<T> {
public:
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        const auto& x = inputs[0];
        const auto& y = inputs[1];

        saveForBackward("x", x);
        saveForBackward("y", y);

        return div(x, y);
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto x = getSavedTensor("x");
        auto y = getSavedTensor("y");

        // grad_x = grad_output / y
        auto grad_x = div(grad_output, y);

        // grad_y = -grad_output * x / (y * y)
        auto y_squared = mul(y, y);
        auto grad_y = div(mul(grad_output, x), y_squared);
        grad_y = mul(grad_y, Tensor<T>({static_cast<T>(-1)}));  // Negate

        grad_x = ops::sumToShape(grad_x, x.shape());
        grad_y = ops::sumToShape(grad_y, y.shape());

        return {grad_x, grad_y};
    }

    std::string name() const override { return "DivOperation"; }
};
```

**数値安定性**:
- `y = 0` の場合、forward で無限大になる（C++ の `/` 演算子の挙動に従う）
- backward で `y = 0` の場合も同様に無限大になる
- 現段階では明示的なエラーチェックは行わず、IEEE 754 の挙動に従う

### Phase 3: MatMul Operation

#### 3.1 MatMulOperation

**数学的定義**:
- Forward: `Z = X @ Y` (行列積)
  - `X`: `[M, K]`
  - `Y`: `[K, N]`
  - `Z`: `[M, N]`
- Backward:
  - `∂L/∂X = ∂L/∂Z @ Y^T` → `[M, N] @ [N, K] = [M, K]`
  - `∂L/∂Y = X^T @ ∂L/∂Z` → `[K, M] @ [M, N] = [K, N]`

**実装**:
```cpp
template <typename T>
class MatMulOperation : public Operation<T> {
public:
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 2) {
            throw std::invalid_argument("MatMulOperation requires exactly 2 inputs");
        }

        const auto& x = inputs[0];
        const auto& y = inputs[1];

        if (x.ndim() != 2 || y.ndim() != 2) {
            throw std::invalid_argument("MatMulOperation requires 2D tensors");
        }

        // Save inputs for backward
        saveForBackward("x", x);
        saveForBackward("y", y);

        return matmul(x, y);
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto x = getSavedTensor("x");
        auto y = getSavedTensor("y");

        // grad_x = grad_output @ y^T
        auto y_T = y.transpose(0, 1);
        auto grad_x = matmul(grad_output, y_T);

        // grad_y = x^T @ grad_output
        auto x_T = x.transpose(0, 1);
        auto grad_y = matmul(x_T, grad_output);

        return {grad_x, grad_y};
    }

    std::string name() const override { return "MatMulOperation"; }
};
```

**注意点**:
- MatMul では Broadcasting は発生しない（2D テンソルのみ）
- Transpose 操作が必要（`Tensor::transpose()` が実装済みであることを前提）

### Phase 4: Element-wise Unary Operations

#### 4.1 ExpOperation

**数学的定義**:
- Forward: `z = exp(x)`
- Backward: `∂L/∂x = ∂L/∂z * exp(x) = ∂L/∂z * z`

**実装**:
```cpp
template <typename T>
class ExpOperation : public Operation<T> {
public:
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("ExpOperation requires exactly 1 input");
        }

        auto result = exp(inputs[0]);

        // Save output for backward (more efficient than recomputing exp)
        saveForBackward("output", result);

        return result;
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto output = getSavedTensor("output");

        // grad_x = grad_output * output (since output = exp(x))
        auto grad_x = mul(grad_output, output);

        return {grad_x};
    }

    std::string name() const override { return "ExpOperation"; }
};
```

**最適化**:
- backward で `exp(x)` を再計算するのではなく、forward の結果を保存して再利用

#### 4.2 LogOperation

**数学的定義**:
- Forward: `z = log(x)`
- Backward: `∂L/∂x = ∂L/∂z / x`

**実装**:
```cpp
template <typename T>
class LogOperation : public Operation<T> {
public:
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("LogOperation requires exactly 1 input");
        }

        const auto& x = inputs[0];

        // Save input for backward
        saveForBackward("x", x);

        return log(x);
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto x = getSavedTensor("x");

        // grad_x = grad_output / x
        auto grad_x = div(grad_output, x);

        return {grad_x};
    }

    std::string name() const override { return "LogOperation"; }
};
```

**数値安定性**:
- `x <= 0` の場合、`log(x)` は `-inf` または `NaN` になる
- 現段階では明示的なチェックは行わず、標準ライブラリの挙動に従う

#### 4.3 SqrtOperation

**数学的定義**:
- Forward: `z = sqrt(x)`
- Backward: `∂L/∂x = ∂L/∂z / (2 * z) = ∂L/∂z / (2 * sqrt(x))`

**実装**:
```cpp
template <typename T>
class SqrtOperation : public Operation<T> {
public:
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("SqrtOperation requires exactly 1 input");
        }

        auto result = sqrt(inputs[0]);

        // Save output for backward
        saveForBackward("output", result);

        return result;
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto output = getSavedTensor("output");

        // grad_x = grad_output / (2 * output)
        auto two = Tensor<T>({static_cast<T>(2)});
        auto denominator = mul(two, output);
        auto grad_x = div(grad_output, denominator);

        return {grad_x};
    }

    std::string name() const override { return "SqrtOperation"; }
};
```

#### 4.4 PowOperation

**数学的定義**:
- Forward: `z = x ^ y`
- Backward:
  - `∂L/∂x = ∂L/∂z * y * x^(y-1)`
  - `∂L/∂y = ∂L/∂z * z * log(x)` (if y is a variable)

**実装**:
```cpp
template <typename T>
class PowOperation : public Operation<T> {
public:
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 2) {
            throw std::invalid_argument("PowOperation requires exactly 2 inputs");
        }

        const auto& x = inputs[0];
        const auto& y = inputs[1];

        saveForBackward("x", x);
        saveForBackward("y", y);

        auto result = pow(x, y);
        saveForBackward("output", result);

        return result;
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto x = getSavedTensor("x");
        auto y = getSavedTensor("y");
        auto output = getSavedTensor("output");

        // grad_x = grad_output * y * x^(y-1)
        auto one = Tensor<T>({static_cast<T>(1)});
        auto y_minus_1 = sub(y, one);
        auto x_pow_y_minus_1 = pow(x, y_minus_1);
        auto grad_x = mul(mul(grad_output, y), x_pow_y_minus_1);

        // grad_y = grad_output * output * log(x)
        auto log_x = log(x);
        auto grad_y = mul(mul(grad_output, output), log_x);

        // Handle broadcasting
        grad_x = ops::sumToShape(grad_x, x.shape());
        grad_y = ops::sumToShape(grad_y, y.shape());

        return {grad_x, grad_y};
    }

    std::string name() const override { return "PowOperation"; }
};
```

**数値安定性**:
- `x <= 0` かつ `y` が整数でない場合、`x^y` は複素数になる
- `log(x)` で `x <= 0` の場合、`-inf` または `NaN` になる

---

## テスト戦略

### テストの種類

#### 1. Forward Pass テスト
各演算の forward が正しく動作することを確認する。

```cpp
TEST(OpsGradTest, AddForward) {
    auto x = Tensor<float>({1.0F, 2.0F, 3.0F});
    auto y = Tensor<float>({4.0F, 5.0F, 6.0F});

    auto op = std::make_shared<AddOperation<float>>();
    auto result = op->forward({x, y});

    EXPECT_EQ(result.shape(), Shape({3}));
    EXPECT_FLOAT_EQ(result[{0}], 5.0F);
    EXPECT_FLOAT_EQ(result[{1}], 7.0F);
    EXPECT_FLOAT_EQ(result[{2}], 9.0F);
}
```

#### 2. Backward Pass テスト
各演算の backward で勾配が正しく計算されることを確認する。

```cpp
TEST(OpsGradTest, AddBackward) {
    auto x = Tensor<float>({1.0F, 2.0F, 3.0F});
    auto y = Tensor<float>({4.0F, 5.0F, 6.0F});

    auto op = std::make_shared<AddOperation<float>>();
    auto result = op->forward({x, y});

    auto grad_output = Tensor<float>({1.0F, 1.0F, 1.0F});
    auto grads = op->backward(grad_output);

    ASSERT_EQ(grads.size(), 2);

    // For addition, gradients are passed through
    EXPECT_EQ(grads[0].shape(), x.shape());
    EXPECT_FLOAT_EQ(grads[0][{0}], 1.0F);
    EXPECT_FLOAT_EQ(grads[0][{1}], 1.0F);
    EXPECT_FLOAT_EQ(grads[0][{2}], 1.0F);

    EXPECT_EQ(grads[1].shape(), y.shape());
    EXPECT_FLOAT_EQ(grads[1][{0}], 1.0F);
    EXPECT_FLOAT_EQ(grads[1][{1}], 1.0F);
    EXPECT_FLOAT_EQ(grads[1][{2}], 1.0F);
}
```

#### 3. 数値勾配チェック
自動微分の正しさを数値微分と比較して検証する。

```cpp
TEST(OpsGradTest, MulNumericalGradient) {
    auto x = Tensor<float>({2.0F, 3.0F});
    auto y = Tensor<float>({4.0F, 5.0F});

    auto op = std::make_shared<MulOperation<float>>();

    bool passed = ops::test::checkNumericalGradient(*op, {x, y});
    EXPECT_TRUE(passed);
}
```

**数値勾配の計算方法**:
```
∂f/∂xi ≈ (f(x + εei) - f(x - εei)) / (2ε)
```
ここで、`ei` は i 番目の要素のみが 1 で他が 0 のベクトル。

#### 4. Broadcasting テスト
Broadcasting が発生する場合に勾配が正しく調整されることを確認する。

```cpp
TEST(OpsGradTest, AddBroadcastBackward) {
    auto x = Tensor<float>({{1.0F, 2.0F}, {3.0F, 4.0F}});  // [2, 2]
    auto y = Tensor<float>({10.0F, 20.0F});                 // [2]

    auto op = std::make_shared<AddOperation<float>>();
    auto result = op->forward({x, y});  // Result: [2, 2]

    auto grad_output = Tensor<float>({{1.0F, 1.0F}, {1.0F, 1.0F}});
    auto grads = op->backward(grad_output);

    // grad_x should be [2, 2]
    EXPECT_EQ(grads[0].shape(), x.shape());

    // grad_y should be [2], summed over axis 0
    EXPECT_EQ(grads[1].shape(), y.shape());
    EXPECT_FLOAT_EQ(grads[1][{0}], 2.0F);  // Sum of [1.0, 1.0]
    EXPECT_FLOAT_EQ(grads[1][{1}], 2.0F);  // Sum of [1.0, 1.0]
}
```

### テストカバレッジ目標
- **行カバレッジ**: > 95%
- **ブランチカバレッジ**: > 90%
- **すべての Operation クラス**: forward, backward, エラーケース

---

## 実装タスクリスト

### Phase 1: 共通ユーティリティ（優先度: 高）
- [ ] `include/gradflow/autograd/ops/op_utils.hpp` を作成
  - [ ] `sumToShape()` 関数を実装
  - [ ] Broadcasting 勾配調整のロジック
  - [ ] Doxygen コメントを追加
- [ ] `tests/test_ops_grad.cpp` を作成（テストファイルの骨組み）
  - [ ] `checkNumericalGradient()` 関数を実装
  - [ ] テストフィクスチャ `OpsGradTest` を定義

### Phase 2: Element-wise Binary Operations（優先度: 高）
- [ ] `include/gradflow/autograd/ops/add.hpp` を作成
  - [ ] `AddOperation` クラスを実装
  - [ ] Doxygen コメントを追加
- [ ] `tests/test_ops_grad.cpp` に `AddGradient` テストを追加
  - [ ] Forward テスト
  - [ ] Backward テスト
  - [ ] Broadcasting テスト
  - [ ] 数値勾配チェック

- [ ] `include/gradflow/autograd/ops/sub.hpp` を作成
  - [ ] `SubOperation` クラスを実装
- [ ] `tests/test_ops_grad.cpp` に `SubGradient` テストを追加

- [ ] `include/gradflow/autograd/ops/mul.hpp` を作成
  - [ ] `MulOperation` クラスを実装
- [ ] `tests/test_ops_grad.cpp` に `MulGradient` テストを追加

- [ ] `include/gradflow/autograd/ops/div.hpp` を作成
  - [ ] `DivOperation` クラスを実装
- [ ] `tests/test_ops_grad.cpp` に `DivGradient` テストを追加

### Phase 3: MatMul Operation（優先度: 高）
- [ ] `include/gradflow/autograd/ops/matmul_op.hpp` を作成
  - [ ] `MatMulOperation` クラスを実装
  - [ ] Transpose を使用した勾配計算
- [ ] `tests/test_ops_grad.cpp` に `MatMulGradient` テストを追加
  - [ ] 2D × 2D のテスト
  - [ ] 長方行列のテスト
  - [ ] 数値勾配チェック

### Phase 4: Element-wise Unary Operations（優先度: 中）
- [ ] `include/gradflow/autograd/ops/exp.hpp` を作成
  - [ ] `ExpOperation` クラスを実装
- [ ] `tests/test_ops_grad.cpp` に `ExpGradient` テストを追加

- [ ] `include/gradflow/autograd/ops/log.hpp` を作成
  - [ ] `LogOperation` クラスを実装
- [ ] `tests/test_ops_grad.cpp` に `LogGradient` テストを追加

- [ ] `include/gradflow/autograd/ops/sqrt.hpp` を作成
  - [ ] `SqrtOperation` クラスを実装
- [ ] `tests/test_ops_grad.cpp` に `SqrtGradient` テストを追加

- [ ] `include/gradflow/autograd/ops/pow.hpp` を作成
  - [ ] `PowOperation` クラスを実装
- [ ] `tests/test_ops_grad.cpp` に `PowGradient` テストを追加

### Phase 5: 統合と最終チェック（優先度: 高）
- [ ] すべてのテストが pass することを確認
- [ ] Clang-Tidy チェックを実行し、警告をゼロにする
- [ ] コードカバレッジを計測し、> 95% を達成
- [ ] CMakeLists.txt に新しいテストファイルを追加
- [ ] ドキュメントを更新（該当する場合）

---

## 完了基準

### 機能要件
- ✅ 9 つの Operation クラスがすべて実装されている
- ✅ 各 Operation で forward と backward が正しく動作する
- ✅ Broadcasting に対応した勾配計算が実装されている

### テスト要件
- ✅ すべてのテストが pass（0 failures）
- ✅ 数値勾配チェックがすべて pass（誤差 < 1e-3）
- ✅ コードカバレッジ > 95%

### コード品質要件
- ✅ Clang-Tidy の警告がゼロ
- ✅ Clang-Format が適用されている
- ✅ Doxygen コメントがすべてのクラスと関数に記載されている
- ✅ SOLID 原則に従った設計

### CI/CD 要件
- ✅ すべての CI チェックが pass
  - ✅ Build & Test (Linux, macOS)
  - ✅ Clang-Tidy
  - ✅ Code Format Check
  - ✅ Sanitizers (AddressSanitizer, UndefinedBehaviorSanitizer)

---

## 実装の優先順位と依存関係

```
Phase 1: op_utils.hpp (必須)
    ↓
Phase 2: Element-wise Binary Ops (並行可能)
    ├── AddOperation
    ├── SubOperation
    ├── MulOperation
    └── DivOperation
    ↓
Phase 3: MatMulOperation (Phase 2 に依存しない)
    ↓
Phase 4: Unary Ops (並行可能)
    ├── ExpOperation
    ├── LogOperation
    ├── SqrtOperation
    └── PowOperation
    ↓
Phase 5: 統合と最終チェック
```

**推奨実装順序**:
1. `op_utils.hpp` と `test_ops_grad.cpp` の骨組み
2. `AddOperation` （最もシンプル）
3. `MulOperation` （入力の保存が必要）
4. `SubOperation`, `DivOperation`
5. `MatMulOperation` （独立しているため並行可能）
6. `ExpOperation`, `LogOperation`, `SqrtOperation`
7. `PowOperation` （最も複雑）
8. 統合テストと最終チェック

---

## 参考資料

### 自動微分の理論
- [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)
- [Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/)

### 実装参考
- [PyTorch Autograd Implementation](https://github.com/pytorch/pytorch/tree/main/torch/csrc/autograd)
- [TinyGrad Operations](https://github.com/tinygrad/tinygrad/tree/master/tinygrad/ops)

### Broadcasting の仕様
- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)

---

**設計レビュー完了日**: 2025-12-31
**設計者**: ml-lib-architect (AI Agent)
**レビュワー**: ml-project-conductor
