# Issue #10: 活性化関数 - 実装ファイル構成

## 1. 概要

7 つの活性化関数（ReLU, Sigmoid, Tanh, GELU, LeakyReLU, Softmax, LogSoftmax）を実装するファイル構成を定義します。既存の Operation 実装パターンに従い、各活性化関数を個別のヘッダーファイルで実装します。

## 2. ディレクトリ構造

```
gradflow/
├── include/
│   └── gradflow/
│       ├── autograd/
│       │   ├── operation.hpp              # 既存: Operation 基底クラス
│       │   └── ops/
│       │       ├── elementwise.hpp        # 既存: 要素ごとの演算ヘルパー
│       │       ├── reduction.hpp          # 既存: reduction 演算
│       │       ├── op_utils.hpp           # 既存: ユーティリティ関数
│       │       ├── relu.hpp               # 新規: ReLU
│       │       ├── sigmoid.hpp            # 新規: Sigmoid
│       │       ├── tanh.hpp               # 新規: Tanh
│       │       ├── gelu.hpp               # 新規: GELU
│       │       ├── leaky_relu.hpp         # 新規: LeakyReLU
│       │       ├── softmax.hpp            # 新規: Softmax
│       │       └── log_softmax.hpp        # 新規: LogSoftmax
│       └── tensor.hpp                     # 既存: Tensor クラス
├── tests/
│   ├── test_activation_ops.cpp            # 新規: 活性化関数の基本テスト
│   └── test_activation_grad.cpp           # 新規: 数値勾配チェック
└── docs/
    ├── ISSUE_10_DESIGN.md                 # 本設計書
    ├── ISSUE_10_TEST_STRATEGY.md          # テスト戦略
    └── ISSUE_10_IMPLEMENTATION_STRUCTURE.md  # 本ファイル
```

## 3. 各ヘッダーファイルの詳細

### 3.1 include/gradflow/autograd/ops/relu.hpp

**目的**: ReLU (Rectified Linear Unit) の実装

**内容**:
```cpp
#pragma once

#include "../operation.hpp"
#include "../tensor.hpp"
#include "elementwise.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gradflow {

/**
 * @brief ReLU (Rectified Linear Unit) operation
 *
 * Applies the rectified linear unit function element-wise.
 *
 * Forward:
 *   y = max(0, x)
 *
 * Backward:
 *   ∂L/∂x = ∂L/∂y * (x > 0 ? 1 : 0)
 *
 * Properties:
 * - Non-saturating activation function
 * - Widely used in deep learning
 * - Simple and fast to compute
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class ReLUOperation : public Operation<T> {
public:
    /**
     * @brief Forward pass: apply ReLU function
     *
     * @param inputs Vector of input tensors (size must be 1)
     * @return Output tensor with ReLU applied
     * @throws std::invalid_argument if inputs size is not 1
     */
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("ReLUOperation requires exactly 1 input");
        }

        const auto& x = inputs[0];

        // y = max(0, x)
        auto result = max(x, static_cast<T>(0));

        // Save input for backward
        this->saveForBackward("input", x);

        return result;
    }

    /**
     * @brief Backward pass: compute gradient
     *
     * @param grad_output Gradient of loss with respect to output
     * @return Vector of gradients with respect to input
     */
    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto input = this->getSavedTensor("input");

        // grad_x = grad_output * (x > 0 ? 1 : 0)
        // Create mask: 1 where input > 0, 0 otherwise
        Tensor<T> mask(input.shape());
        for (size_t i = 0; i < input.size(); ++i) {
            mask.data()[i] = input.data()[i] > T(0) ? T(1) : T(0);
        }

        auto grad_x = mul(grad_output, mask);

        return {grad_x};
    }

    [[nodiscard]] std::string name() const override { return "ReLUOperation"; }
};

}  // namespace gradflow
```

**依存関係**:
- `operation.hpp`: Operation 基底クラス
- `tensor.hpp`: Tensor クラス
- `elementwise.hpp`: `max`, `mul` ヘルパー関数

**テスト**: `tests/test_activation_ops.cpp`

---

### 3.2 include/gradflow/autograd/ops/sigmoid.hpp

**目的**: Sigmoid 活性化関数の実装

**内容**:
```cpp
#pragma once

#include "../operation.hpp"
#include "../tensor.hpp"
#include "elementwise.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gradflow {

/**
 * @brief Sigmoid activation function
 *
 * Applies the sigmoid function element-wise.
 *
 * Forward:
 *   y = 1 / (1 + exp(-x))
 *
 * Backward:
 *   ∂L/∂x = ∂L/∂y * y * (1 - y)
 *
 * Properties:
 * - Output range: [0, 1]
 * - Non-linear and differentiable
 * - Can suffer from vanishing gradients for large |x|
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class SigmoidOperation : public Operation<T> {
public:
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("SigmoidOperation requires exactly 1 input");
        }

        const auto& x = inputs[0];

        // y = 1 / (1 + exp(-x))
        auto neg_x = mul(x, static_cast<T>(-1));
        auto exp_neg_x = exp(neg_x);
        auto one_plus_exp = add(exp_neg_x, static_cast<T>(1));
        auto result = div(static_cast<T>(1), one_plus_exp);

        // Save output for backward (more efficient than recomputing)
        this->saveForBackward("output", result);

        return result;
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto output = this->getSavedTensor("output");

        // grad_x = grad_output * output * (1 - output)
        auto one_minus_output = sub(static_cast<T>(1), output);
        auto output_times_one_minus_output = mul(output, one_minus_output);
        auto grad_x = mul(grad_output, output_times_one_minus_output);

        return {grad_x};
    }

    [[nodiscard]] std::string name() const override { return "SigmoidOperation"; }
};

}  // namespace gradflow
```

**依存関係**:
- `operation.hpp`: Operation 基底クラス
- `tensor.hpp`: Tensor クラス
- `elementwise.hpp`: `exp`, `mul`, `add`, `sub`, `div` ヘルパー関数

**テスト**: `tests/test_activation_ops.cpp`

---

### 3.3 include/gradflow/autograd/ops/tanh.hpp

**目的**: Tanh (Hyperbolic Tangent) 活性化関数の実装

**内容**:
```cpp
#pragma once

#include "../operation.hpp"
#include "../tensor.hpp"
#include "elementwise.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gradflow {

/**
 * @brief Tanh (Hyperbolic Tangent) activation function
 *
 * Applies the hyperbolic tangent function element-wise.
 *
 * Forward:
 *   y = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 *
 * Backward:
 *   ∂L/∂x = ∂L/∂y * (1 - y²)
 *
 * Properties:
 * - Output range: [-1, 1]
 * - Zero-centered (unlike sigmoid)
 * - Can suffer from vanishing gradients for large |x|
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class TanhOperation : public Operation<T> {
public:
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("TanhOperation requires exactly 1 input");
        }

        const auto& x = inputs[0];

        // Use Tensor-level tanh if available, otherwise compute manually
        auto result = tanh(x);

        // Save output for backward
        this->saveForBackward("output", result);

        return result;
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto output = this->getSavedTensor("output");

        // grad_x = grad_output * (1 - output²)
        auto output_squared = mul(output, output);
        auto one_minus_output_squared = sub(static_cast<T>(1), output_squared);
        auto grad_x = mul(grad_output, one_minus_output_squared);

        return {grad_x};
    }

    [[nodiscard]] std::string name() const override { return "TanhOperation"; }
};

}  // namespace gradflow
```

**依存関係**:
- `operation.hpp`: Operation 基底クラス
- `tensor.hpp`: Tensor クラス
- `elementwise.hpp`: `tanh`, `mul`, `sub` ヘルパー関数

**注意**: Tensor レベルの `tanh` 関数が必要です。実装されていない場合は Phase 1 で追加します。

**テスト**: `tests/test_activation_ops.cpp`

---

### 3.4 include/gradflow/autograd/ops/gelu.hpp

**目的**: GELU (Gaussian Error Linear Unit) 活性化関数の実装

**内容**:
```cpp
#pragma once

#include "../operation.hpp"
#include "../tensor.hpp"
#include "elementwise.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gradflow {

/**
 * @brief GELU (Gaussian Error Linear Unit) activation function
 *
 * Applies the Gaussian Error Linear Unit function element-wise.
 * This implementation uses the tanh approximation for efficiency.
 *
 * Forward (approximate):
 *   y = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 *
 * Backward:
 *   ∂L/∂x = ∂L/∂y * ∂y/∂x
 *   where ∂y/∂x is computed using the chain rule
 *
 * Properties:
 * - Smooth activation function
 * - Recommended for transformer architectures
 * - Provides better gradients than ReLU in some cases
 *
 * References:
 * - Original paper: "Gaussian Error Linear Units (GELUs)"
 * - PyTorch implementation: torch.nn.GELU(approximate='tanh')
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class GELUOperation : public Operation<T> {
public:
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("GELUOperation requires exactly 1 input");
        }

        const auto& x = inputs[0];

        // Constants
        constexpr T sqrt_2_over_pi = static_cast<T>(0.7978845608028654);  // √(2/π)
        constexpr T coeff = static_cast<T>(0.044715);

        // Compute: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        auto x_cubed = mul(mul(x, x), x);
        auto coeff_x_cubed = mul(x_cubed, coeff);
        auto inner = add(x, coeff_x_cubed);
        auto scaled_inner = mul(inner, sqrt_2_over_pi);
        auto tanh_value = tanh(scaled_inner);
        auto one_plus_tanh = add(tanh_value, static_cast<T>(1));
        auto half_x = mul(x, static_cast<T>(0.5));
        auto result = mul(half_x, one_plus_tanh);

        // Save for backward
        this->saveForBackward("input", x);
        this->saveForBackward("tanh_value", tanh_value);

        return result;
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto input = this->getSavedTensor("input");
        auto tanh_value = this->getSavedTensor("tanh_value");

        // Constants
        constexpr T sqrt_2_over_pi = static_cast<T>(0.7978845608028654);  // √(2/π)
        constexpr T coeff = static_cast<T>(0.044715);

        // Compute ∂y/∂x
        // cdf = 0.5 * (1 + tanh(...))
        auto one_plus_tanh = add(tanh_value, static_cast<T>(1));
        auto cdf = mul(one_plus_tanh, static_cast<T>(0.5));

        // pdf_approximation = (1 - tanh²) * √(2/π) * (1 + 3 * 0.044715 * x²)
        auto tanh_squared = mul(tanh_value, tanh_value);
        auto one_minus_tanh_squared = sub(static_cast<T>(1), tanh_squared);
        auto x_squared = mul(input, input);
        auto three_coeff = static_cast<T>(3) * coeff;
        auto three_coeff_x_squared = mul(x_squared, three_coeff);
        auto one_plus_three_coeff_x_squared = add(static_cast<T>(1), three_coeff_x_squared);
        auto pdf_part = mul(one_minus_tanh_squared, sqrt_2_over_pi);
        auto pdf_approximation = mul(pdf_part, one_plus_three_coeff_x_squared);

        // ∂y/∂x = cdf + 0.5 * x * pdf_approximation
        auto half_x = mul(input, static_cast<T>(0.5));
        auto half_x_pdf = mul(half_x, pdf_approximation);
        auto dy_dx = add(cdf, half_x_pdf);

        // grad_x = grad_output * dy_dx
        auto grad_x = mul(grad_output, dy_dx);

        return {grad_x};
    }

    [[nodiscard]] std::string name() const override { return "GELUOperation"; }
};

}  // namespace gradflow
```

**依存関係**:
- `operation.hpp`: Operation 基底クラス
- `tensor.hpp`: Tensor クラス
- `elementwise.hpp`: `tanh`, `mul`, `add`, `sub` ヘルパー関数

**テスト**: `tests/test_activation_ops.cpp`

---

### 3.5 include/gradflow/autograd/ops/leaky_relu.hpp

**目的**: LeakyReLU 活性化関数の実装

**内容**:
```cpp
#pragma once

#include "../operation.hpp"
#include "../tensor.hpp"
#include "elementwise.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gradflow {

/**
 * @brief LeakyReLU activation function
 *
 * Applies the leaky rectified linear unit function element-wise.
 *
 * Forward:
 *   y = x if x > 0 else alpha * x
 *
 * Backward:
 *   ∂L/∂x = ∂L/∂y * (1 if x > 0 else alpha)
 *
 * Properties:
 * - Allows small gradient for negative values
 * - Prevents "dying ReLU" problem
 * - Parameterized by alpha (default: 0.01)
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class LeakyReLUOperation : public Operation<T> {
public:
    /**
     * @brief Constructor
     *
     * @param alpha Slope for negative values (default: 0.01)
     */
    explicit LeakyReLUOperation(T alpha = static_cast<T>(0.01)) : alpha_(alpha) {}

    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("LeakyReLUOperation requires exactly 1 input");
        }

        const auto& x = inputs[0];

        // y = x > 0 ? x : alpha * x
        Tensor<T> result(x.shape());
        for (size_t i = 0; i < x.size(); ++i) {
            result.data()[i] = x.data()[i] > T(0) ? x.data()[i] : alpha_ * x.data()[i];
        }

        // Save input for backward
        this->saveForBackward("input", x);

        return result;
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto input = this->getSavedTensor("input");

        // grad_x = grad_output * (x > 0 ? 1 : alpha)
        Tensor<T> mask(input.shape());
        for (size_t i = 0; i < input.size(); ++i) {
            mask.data()[i] = input.data()[i] > T(0) ? T(1) : alpha_;
        }

        auto grad_x = mul(grad_output, mask);

        return {grad_x};
    }

    [[nodiscard]] std::string name() const override { return "LeakyReLUOperation"; }

private:
    T alpha_;  ///< Slope for negative values
};

}  // namespace gradflow
```

**依存関係**:
- `operation.hpp`: Operation 基底クラス
- `tensor.hpp`: Tensor クラス
- `elementwise.hpp`: `mul` ヘルパー関数

**テスト**: `tests/test_activation_ops.cpp`

---

### 3.6 include/gradflow/autograd/ops/softmax.hpp

**目的**: Softmax 活性化関数の実装（数値安定版）

**内容**:
```cpp
#pragma once

#include "../operation.hpp"
#include "../tensor.hpp"
#include "elementwise.hpp"
#include "reduction.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gradflow {

/**
 * @brief Softmax activation function (numerically stable version)
 *
 * Applies the softmax function along the specified dimension.
 * Uses the log-sum-exp trick for numerical stability.
 *
 * Forward (numerically stable):
 *   max_x = max(x, axis=dim)
 *   exp_shifted = exp(x - max_x)
 *   y = exp_shifted / sum(exp_shifted, axis=dim)
 *
 * Backward:
 *   ∂L/∂x = y * (∂L/∂y - Σ(∂L/∂y * y))
 *
 * Properties:
 * - Output is a probability distribution (sum to 1)
 * - Numerically stable (prevents overflow/underflow)
 * - Used for multi-class classification
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class SoftmaxOperation : public Operation<T> {
public:
    /**
     * @brief Constructor
     *
     * @param dim Dimension along which to apply softmax (default: -1, last dim)
     */
    explicit SoftmaxOperation(int dim = -1) : dim_(dim) {}

    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("SoftmaxOperation requires exactly 1 input");
        }

        const auto& x = inputs[0];

        // Normalize dim
        int actual_dim = dim_ < 0 ? static_cast<int>(x.ndim()) + dim_ : dim_;
        if (actual_dim < 0 || actual_dim >= static_cast<int>(x.ndim())) {
            throw std::invalid_argument("Invalid dimension for softmax");
        }

        // Numerically stable softmax using log-sum-exp trick
        // Step 1: Subtract max for numerical stability
        auto max_x = max(x, actual_dim, /*keepdim=*/true);
        auto x_shifted = sub(x, max_x);

        // Step 2: Compute exp(x - max_x)
        auto exp_shifted = exp(x_shifted);

        // Step 3: Compute sum along the specified dimension
        auto sum_exp = sum(exp_shifted, actual_dim, /*keepdim=*/true);

        // Step 4: Normalize
        auto result = div(exp_shifted, sum_exp);

        // Save output for backward
        this->saveForBackward("output", result);

        return result;
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto output = this->getSavedTensor("output");

        // grad_x = y * (grad_output - Σ(grad_output * y))
        // Step 1: Compute grad_output * y
        auto grad_times_output = mul(grad_output, output);

        // Step 2: Sum along the softmax dimension
        int actual_dim = dim_ < 0 ? static_cast<int>(output.ndim()) + dim_ : dim_;
        auto sum_grad_times_output = sum(grad_times_output, actual_dim, /*keepdim=*/true);

        // Step 3: Subtract from grad_output
        auto grad_minus_sum = sub(grad_output, sum_grad_times_output);

        // Step 4: Multiply by output
        auto grad_x = mul(output, grad_minus_sum);

        return {grad_x};
    }

    [[nodiscard]] std::string name() const override { return "SoftmaxOperation"; }

private:
    int dim_;  ///< Dimension along which to apply softmax
};

}  // namespace gradflow
```

**依存関係**:
- `operation.hpp`: Operation 基底クラス
- `tensor.hpp`: Tensor クラス
- `elementwise.hpp`: `exp`, `mul`, `sub`, `div` ヘルパー関数
- `reduction.hpp`: `sum`, `max` ヘルパー関数

**注意**: `max(tensor, dim, keepdim)` と `sum(tensor, dim, keepdim)` が必要です。

**テスト**: `tests/test_activation_ops.cpp`

---

### 3.7 include/gradflow/autograd/ops/log_softmax.hpp

**目的**: LogSoftmax 活性化関数の実装（数値安定版）

**内容**:
```cpp
#pragma once

#include "../operation.hpp"
#include "../tensor.hpp"
#include "elementwise.hpp"
#include "reduction.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gradflow {

/**
 * @brief LogSoftmax activation function (numerically stable version)
 *
 * Applies the log softmax function along the specified dimension.
 * Uses the log-sum-exp trick for numerical stability.
 *
 * Forward (numerically stable):
 *   max_x = max(x, axis=dim)
 *   log_sum_exp = max_x + log(sum(exp(x - max_x), axis=dim))
 *   y = x - log_sum_exp
 *
 * Backward:
 *   ∂L/∂x = ∂L/∂y - sum(∂L/∂y, axis=dim) * exp(y)
 *
 * Properties:
 * - Computes log probabilities
 * - Numerically stable (prevents overflow/underflow)
 * - Often used with NLLLoss for classification
 *
 * @tparam T Element type (float, double, etc.)
 */
template <typename T>
class LogSoftmaxOperation : public Operation<T> {
public:
    /**
     * @brief Constructor
     *
     * @param dim Dimension along which to apply log softmax (default: -1, last dim)
     */
    explicit LogSoftmaxOperation(int dim = -1) : dim_(dim) {}

    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        if (inputs.size() != 1) {
            throw std::invalid_argument("LogSoftmaxOperation requires exactly 1 input");
        }

        const auto& x = inputs[0];

        // Normalize dim
        int actual_dim = dim_ < 0 ? static_cast<int>(x.ndim()) + dim_ : dim_;
        if (actual_dim < 0 || actual_dim >= static_cast<int>(x.ndim())) {
            throw std::invalid_argument("Invalid dimension for log_softmax");
        }

        // Numerically stable log_softmax using log-sum-exp trick
        // Step 1: Subtract max for numerical stability
        auto max_x = max(x, actual_dim, /*keepdim=*/true);
        auto x_shifted = sub(x, max_x);

        // Step 2: Compute log(sum(exp(x - max_x)))
        auto exp_shifted = exp(x_shifted);
        auto sum_exp = sum(exp_shifted, actual_dim, /*keepdim=*/true);
        auto log_sum_exp = log(sum_exp);

        // Step 3: Add back max_x to get log(sum(exp(x)))
        auto log_sum_exp_original = add(max_x, log_sum_exp);

        // Step 4: Subtract from original x
        auto result = sub(x, log_sum_exp_original);

        // Save output for backward
        this->saveForBackward("output", result);

        return result;
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto output = this->getSavedTensor("output");

        // grad_x = grad_output - sum(grad_output, axis=dim) * exp(output)
        // Step 1: Sum grad_output along the dimension
        int actual_dim = dim_ < 0 ? static_cast<int>(output.ndim()) + dim_ : dim_;
        auto sum_grad = sum(grad_output, actual_dim, /*keepdim=*/true);

        // Step 2: Multiply by exp(output) = softmax
        auto exp_output = exp(output);
        auto sum_grad_times_exp = mul(sum_grad, exp_output);

        // Step 3: Subtract from grad_output
        auto grad_x = sub(grad_output, sum_grad_times_exp);

        return {grad_x};
    }

    [[nodiscard]] std::string name() const override { return "LogSoftmaxOperation"; }

private:
    int dim_;  ///< Dimension along which to apply log softmax
};

}  // namespace gradflow
```

**依存関係**:
- `operation.hpp`: Operation 基底クラス
- `tensor.hpp`: Tensor クラス
- `elementwise.hpp`: `exp`, `log`, `mul`, `sub`, `add` ヘルパー関数
- `reduction.hpp`: `sum`, `max` ヘルパー関数

**テスト**: `tests/test_activation_ops.cpp`

---

### 3.8 include/gradflow/autograd/ops/op_utils.hpp（拡張）

**目的**: 活性化関数で共通して使うヘルパー関数を追加

**追加内容**:
- `max(tensor, dim, keepdim)`: 指定次元の最大値を計算
- `sum(tensor, dim, keepdim)`: 指定次元の総和を計算（既存の `reduction.hpp` を確認）

**注意**: これらの関数が既に実装されている場合は追加不要です。

---

## 4. テストファイルの詳細

### 4.1 tests/test_activation_ops.cpp

**目的**: 活性化関数の forward/backward の基本テスト

**内容**:
- 各活性化関数の forward テスト
- 各活性化関数の backward テスト
- 2D テンソルでのテスト
- パラメータテスト（LeakyReLU の alpha）
- Softmax/LogSoftmax の数値安定性テスト

**テストケース数**: 約 30-40 テスト

**参考**: `tests/test_ops_grad.cpp` の構造を踏襲

---

### 4.2 tests/test_activation_grad.cpp

**目的**: 数値勾配チェック

**内容**:
- 各活性化関数で数値勾配チェックを実行
- 相対誤差が閾値以下であることを確認

**テストケース数**: 約 7 テスト（各活性化関数 1 つずつ）

---

## 5. 実装の優先順位

### Phase 1: 基本的な活性化関数（1-3 日）

**実装順序**:
1. `relu.hpp` + テスト
2. `sigmoid.hpp` + テスト
3. `tanh.hpp` + テスト

**完了基準**: 3 つの活性化関数のすべてのテストが pass

---

### Phase 2: パラメータ付き活性化関数（2-3 日）

**実装順序**:
4. `leaky_relu.hpp` + テスト
5. `gelu.hpp` + テスト

**完了基準**: 5 つの活性化関数のすべてのテストが pass

---

### Phase 3: 正規化系活性化関数（3-4 日）

**実装順序**:
6. Tensor レベルの `max(tensor, dim, keepdim)` と `sum(tensor, dim, keepdim)` を確認・実装
7. `softmax.hpp` + テスト
8. `log_softmax.hpp` + テスト

**完了基準**:
- すべての活性化関数のテストが pass
- Softmax/LogSoftmax の数値安定性テストが pass

---

### Phase 4: 統合テストと検証（1-2 日）

**実装順序**:
9. 数値勾配チェック（`test_activation_grad.cpp`）
10. CI チェック（build, test, clang-tidy, sanitizers, format）
11. ドキュメント整備

**完了基準**: すべての CI チェックが pass

---

## 6. 必要な Tensor レベル演算の確認

以下の演算が Tensor レベルで実装されているかを確認します：

**既存の演算（確認済み）**:
- ✅ `add`, `sub`, `mul`, `div`
- ✅ `exp`, `log`, `sqrt`, `pow`
- ✅ `sum(tensor, axis)` (reduction.hpp)

**確認が必要な演算**:
- ⏳ `tanh(tensor)`: 要素ごとの双曲線正接
- ⏳ `max(tensor, scalar)`: スカラーとの最大値
- ⏳ `max(tensor, axis, keepdim)`: 指定軸での最大値
- ⏳ `sum(tensor, axis, keepdim)`: 指定軸での総和（keepdim オプション）

**対応**:
1. `tanh` が未実装の場合 → Phase 1 で実装
2. `max(tensor, scalar)` が未実装の場合 → `relu.hpp` で直接実装
3. `max/sum` に `keepdim` オプションがない場合 → 追加または reshape で対応

---

## 7. CI チェック対応

### 7.1 Clang-Tidy

**チェック項目**:
- 未使用変数
- const correctness
- 命名規則
- メモリリーク

**対策**: 既存のコードスタイルに従う

---

### 7.2 Sanitizers

**チェック項目**:
- ASAN: メモリリーク、use-after-free
- UBSAN: 未定義動作

**対策**: Tensor のライフタイムに注意

---

### 7.3 Code Format

**チェック項目**:
- clang-format
- cmake-format
- yamllint

**対策**: コミット前に `./scripts/ci-format-apply.sh` を実行

---

## 8. ドキュメント

### 8.1 実装中に作成するドキュメント

- ✅ `docs/ISSUE_10_DESIGN.md`: 技術設計書
- ✅ `docs/ISSUE_10_TEST_STRATEGY.md`: テスト戦略
- ✅ `docs/ISSUE_10_IMPLEMENTATION_STRUCTURE.md`: 本ファイル

### 8.2 実装後に作成するドキュメント

- ⏳ `docs/ACTIVATION_FUNCTIONS.md`: 活性化関数の使用ガイド
- ⏳ `docs/PROGRESS.md`: プロジェクト進捗の更新

---

## 9. 完了基準

以下の条件をすべて満たすことで Issue #10 を完了とします：

- ✅ 7 つの活性化関数がすべて実装されている
- ✅ 各活性化関数の forward と backward が正しく動作する
- ✅ すべての単体テストが pass する
- ✅ すべての数値勾配チェックが pass する（相対誤差 < 1e-2）
- ✅ Softmax/LogSoftmax の数値安定性テストが pass する
- ✅ すべての CI チェックが pass する
  - Build & Test (全プラットフォーム)
  - Clang-Tidy
  - Sanitizers (ASAN, UBSAN)
  - Code Format
- ✅ ドキュメントが整備されている

---

## 10. 次のステップ

Issue #10 完了後、Phase 2.5 へ進みます：

- **Phase 2.5**: 損失関数の実装 (CrossEntropyLoss, MSELoss, BCELoss など)
- **Phase 2.6**: Optimizer の実装 (SGD, Adam など)

---

## 11. 参考資料

### 既存の実装
- `include/gradflow/autograd/operation.hpp`: Operation 基底クラス
- `include/gradflow/autograd/ops/exp.hpp`: Exp の実装例
- `tests/test_ops_grad.cpp`: テストの実装例

### 外部ライブラリ
- [PyTorch Activation Functions](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/activation.py)
- [JAX GELU](https://docs.jax.dev/en/latest/_autosummary/jax.nn.gelu.html)
