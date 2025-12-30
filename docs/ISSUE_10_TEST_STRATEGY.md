# Issue #10: 活性化関数 - テスト戦略

## 1. テスト概要

7 つの活性化関数（ReLU, Sigmoid, Tanh, GELU, LeakyReLU, Softmax, LogSoftmax）に対して、以下の観点から包括的なテストを実施します。

**テストの目的**:
1. 各活性化関数の forward pass が正しく動作することを確認
2. 各活性化関数の backward pass（勾配計算）が正しく動作することを確認
3. 数値勾配チェックで自動微分の精度を検証
4. Softmax/LogSoftmax の数値安定性を確認
5. エッジケースや境界値での動作を検証

## 2. テスト構成

### 2.1 テストファイル

```
tests/
├── test_activation_ops.cpp     # 活性化関数の基本テスト（forward/backward）
├── test_activation_grad.cpp    # 数値勾配チェック
└── test_activation_stability.cpp  # Softmax/LogSoftmax の数値安定性テスト
```

### 2.2 テストクラス

```cpp
class ActivationOpsTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // ヘルパー関数: 浮動小数点の近似比較
    bool approx_equal(float a, float b, float epsilon = 1e-5F) {
        return std::abs(a - b) < epsilon;
    }

    // ヘルパー関数: ベクトルの総和
    float sum_vector(const std::vector<float>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0F);
    }
};
```

## 3. 各活性化関数のテスト詳細

### 3.1 ReLU (Rectified Linear Unit)

#### 3.1.1 Forward テスト

**目的**: ReLU が正の値をそのまま、負の値を 0 に変換することを確認

```cpp
TEST_F(ActivationOpsTest, ReLUForward) {
    // 1D テンソル
    auto x = Tensor<float>({-2.0F, -1.0F, 0.0F, 1.0F, 2.0F});
    auto op = std::make_shared<ReLUOperation<float>>();
    auto result = op->forward({x});

    EXPECT_EQ(result.shape(), Shape({5}));
    EXPECT_FLOAT_EQ(result[{0}], 0.0F);  // -2 -> 0
    EXPECT_FLOAT_EQ(result[{1}], 0.0F);  // -1 -> 0
    EXPECT_FLOAT_EQ(result[{2}], 0.0F);  //  0 -> 0
    EXPECT_FLOAT_EQ(result[{3}], 1.0F);  //  1 -> 1
    EXPECT_FLOAT_EQ(result[{4}], 2.0F);  //  2 -> 2
}
```

#### 3.1.2 Backward テスト

**目的**: ReLU の勾配が正の値で 1、負の値で 0 になることを確認

```cpp
TEST_F(ActivationOpsTest, ReLUBackward) {
    auto x = Tensor<float>({-2.0F, -1.0F, 0.0F, 1.0F, 2.0F});
    auto op = std::make_shared<ReLUOperation<float>>();
    auto result = op->forward({x});

    auto grad_output = Tensor<float>({1.0F, 1.0F, 1.0F, 1.0F, 1.0F});
    auto grads = op->backward(grad_output);

    ASSERT_EQ(grads.size(), 1UL);
    EXPECT_EQ(grads[0].shape(), x.shape());
    EXPECT_FLOAT_EQ(grads[0][{0}], 0.0F);  // x < 0 -> grad = 0
    EXPECT_FLOAT_EQ(grads[0][{1}], 0.0F);  // x < 0 -> grad = 0
    EXPECT_FLOAT_EQ(grads[0][{2}], 0.0F);  // x = 0 -> grad = 0
    EXPECT_FLOAT_EQ(grads[0][{3}], 1.0F);  // x > 0 -> grad = 1
    EXPECT_FLOAT_EQ(grads[0][{4}], 1.0F);  // x > 0 -> grad = 1
}
```

#### 3.1.3 2D テンソルテスト

**目的**: 多次元テンソルでも正しく動作することを確認

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

#### 3.1.4 数値勾配チェック

```cpp
TEST_F(ActivationOpsTest, ReLUNumericalGradient) {
    auto x = Tensor<float>({-1.0F, 0.5F, 1.0F});
    auto op = std::make_shared<ReLUOperation<float>>();

    bool grad_correct = ops::test::checkNumericalGradient(
        *op, {x}, {}, 1e-4F, 1e-2F);
    EXPECT_TRUE(grad_correct);
}
```

### 3.2 Sigmoid

#### 3.2.1 Forward テスト

**目的**: Sigmoid が正しい出力範囲 [0, 1] を持つことを確認

```cpp
TEST_F(ActivationOpsTest, SigmoidForward) {
    auto x = Tensor<float>({-2.0F, 0.0F, 2.0F});
    auto op = std::make_shared<SigmoidOperation<float>>();
    auto result = op->forward({x});

    EXPECT_EQ(result.shape(), Shape({3}));

    // sigmoid(0) = 0.5
    EXPECT_NEAR(result[{1}], 0.5F, 1e-5F);

    // sigmoid(-x) + sigmoid(x) = 1 を確認
    EXPECT_NEAR(result[{0}] + result[{2}], 1.0F, 1e-5F);

    // 出力が [0, 1] の範囲内
    EXPECT_GT(result[{0}], 0.0F);
    EXPECT_LT(result[{0}], 1.0F);
    EXPECT_GT(result[{2}], 0.0F);
    EXPECT_LT(result[{2}], 1.0F);
}
```

#### 3.2.2 Backward テスト

**目的**: Sigmoid の勾配 y * (1 - y) が正しく計算されることを確認

```cpp
TEST_F(ActivationOpsTest, SigmoidBackward) {
    auto x = Tensor<float>({0.0F});
    auto op = std::make_shared<SigmoidOperation<float>>();
    auto output = op->forward({x});

    // sigmoid(0) = 0.5
    EXPECT_NEAR(output[{0}], 0.5F, 1e-5F);

    auto grad_output = Tensor<float>({1.0F});
    auto grads = op->backward(grad_output);

    ASSERT_EQ(grads.size(), 1UL);

    // grad = y * (1 - y) = 0.5 * 0.5 = 0.25
    EXPECT_NEAR(grads[0][{0}], 0.25F, 1e-5F);
}
```

#### 3.2.3 数値勾配チェック

```cpp
TEST_F(ActivationOpsTest, SigmoidNumericalGradient) {
    auto x = Tensor<float>({-1.0F, 0.0F, 1.0F});
    auto op = std::make_shared<SigmoidOperation<float>>();

    bool grad_correct = ops::test::checkNumericalGradient(
        *op, {x}, {}, 1e-4F, 1e-2F);
    EXPECT_TRUE(grad_correct);
}
```

### 3.3 Tanh

#### 3.3.1 Forward テスト

**目的**: Tanh が正しい出力範囲 [-1, 1] を持つことを確認

```cpp
TEST_F(ActivationOpsTest, TanhForward) {
    auto x = Tensor<float>({-2.0F, 0.0F, 2.0F});
    auto op = std::make_shared<TanhOperation<float>>();
    auto result = op->forward({x});

    EXPECT_EQ(result.shape(), Shape({3}));

    // tanh(0) = 0
    EXPECT_NEAR(result[{1}], 0.0F, 1e-5F);

    // tanh(-x) = -tanh(x) を確認
    EXPECT_NEAR(result[{0}], -result[{2}], 1e-5F);

    // 出力が [-1, 1] の範囲内
    EXPECT_GT(result[{0}], -1.0F);
    EXPECT_LT(result[{0}], 1.0F);
    EXPECT_GT(result[{2}], -1.0F);
    EXPECT_LT(result[{2}], 1.0F);
}
```

#### 3.3.2 Backward テスト

**目的**: Tanh の勾配 1 - y² が正しく計算されることを確認

```cpp
TEST_F(ActivationOpsTest, TanhBackward) {
    auto x = Tensor<float>({0.0F});
    auto op = std::make_shared<TanhOperation<float>>();
    auto output = op->forward({x});

    // tanh(0) = 0
    EXPECT_NEAR(output[{0}], 0.0F, 1e-5F);

    auto grad_output = Tensor<float>({1.0F});
    auto grads = op->backward(grad_output);

    ASSERT_EQ(grads.size(), 1UL);

    // grad = 1 - y² = 1 - 0 = 1
    EXPECT_NEAR(grads[0][{0}], 1.0F, 1e-5F);
}
```

#### 3.3.3 数値勾配チェック

```cpp
TEST_F(ActivationOpsTest, TanhNumericalGradient) {
    auto x = Tensor<float>({-1.0F, 0.0F, 1.0F});
    auto op = std::make_shared<TanhOperation<float>>();

    bool grad_correct = ops::test::checkNumericalGradient(
        *op, {x}, {}, 1e-4F, 1e-2F);
    EXPECT_TRUE(grad_correct);
}
```

### 3.4 GELU (Gaussian Error Linear Unit)

#### 3.4.1 Forward テスト

**目的**: GELU が滑らかな非線形性を持つことを確認

```cpp
TEST_F(ActivationOpsTest, GELUForward) {
    auto x = Tensor<float>({-3.0F, 0.0F, 3.0F});
    auto op = std::make_shared<GELUOperation<float>>();
    auto result = op->forward({x});

    EXPECT_EQ(result.shape(), Shape({3}));

    // GELU(0) ≈ 0
    EXPECT_NEAR(result[{1}], 0.0F, 1e-3F);

    // 大きな正の値では GELU(x) ≈ x
    EXPECT_NEAR(result[{2}], 3.0F, 1e-2F);

    // 大きな負の値では GELU(x) ≈ 0
    EXPECT_NEAR(result[{0}], 0.0F, 1e-2F);
}
```

#### 3.4.2 Backward テスト

```cpp
TEST_F(ActivationOpsTest, GELUBackward) {
    auto x = Tensor<float>({0.0F, 1.0F});
    auto op = std::make_shared<GELUOperation<float>>();
    auto output = op->forward({x});

    auto grad_output = Tensor<float>({1.0F, 1.0F});
    auto grads = op->backward(grad_output);

    ASSERT_EQ(grads.size(), 1UL);
    EXPECT_EQ(grads[0].shape(), x.shape());

    // 勾配が有限値であることを確認
    EXPECT_TRUE(std::isfinite(grads[0][{0}]));
    EXPECT_TRUE(std::isfinite(grads[0][{1}]));
}
```

#### 3.4.3 数値勾配チェック

```cpp
TEST_F(ActivationOpsTest, GELUNumericalGradient) {
    auto x = Tensor<float>({-1.0F, 0.0F, 1.0F});
    auto op = std::make_shared<GELUOperation<float>>();

    bool grad_correct = ops::test::checkNumericalGradient(
        *op, {x}, {}, 1e-4F, 5e-2F);  // GELU は近似なので許容誤差を大きく
    EXPECT_TRUE(grad_correct);
}
```

### 3.5 LeakyReLU

#### 3.5.1 Forward テスト（デフォルト alpha）

**目的**: LeakyReLU が負の値に小さな勾配を持つことを確認

```cpp
TEST_F(ActivationOpsTest, LeakyReLUForward) {
    float alpha = 0.01F;
    auto x = Tensor<float>({-2.0F, -1.0F, 0.0F, 1.0F, 2.0F});
    auto op = std::make_shared<LeakyReLUOperation<float>>(alpha);
    auto result = op->forward({x});

    EXPECT_EQ(result.shape(), Shape({5}));
    EXPECT_FLOAT_EQ(result[{0}], -2.0F * alpha);
    EXPECT_FLOAT_EQ(result[{1}], -1.0F * alpha);
    EXPECT_FLOAT_EQ(result[{2}], 0.0F);
    EXPECT_FLOAT_EQ(result[{3}], 1.0F);
    EXPECT_FLOAT_EQ(result[{4}], 2.0F);
}
```

#### 3.5.2 Forward テスト（カスタム alpha）

**目的**: alpha パラメータが正しく動作することを確認

```cpp
TEST_F(ActivationOpsTest, LeakyReLUCustomAlpha) {
    float alpha = 0.2F;
    auto x = Tensor<float>({-1.0F, 1.0F});
    auto op = std::make_shared<LeakyReLUOperation<float>>(alpha);
    auto result = op->forward({x});

    EXPECT_FLOAT_EQ(result[{0}], -1.0F * alpha);
    EXPECT_FLOAT_EQ(result[{1}], 1.0F);
}
```

#### 3.5.3 Backward テスト

```cpp
TEST_F(ActivationOpsTest, LeakyReLUBackward) {
    float alpha = 0.1F;
    auto x = Tensor<float>({-1.0F, 0.0F, 1.0F});
    auto op = std::make_shared<LeakyReLUOperation<float>>(alpha);
    auto result = op->forward({x});

    auto grad_output = Tensor<float>({1.0F, 1.0F, 1.0F});
    auto grads = op->backward(grad_output);

    ASSERT_EQ(grads.size(), 1UL);
    EXPECT_FLOAT_EQ(grads[0][{0}], alpha);  // x < 0 -> grad = alpha
    EXPECT_FLOAT_EQ(grads[0][{1}], alpha);  // x = 0 -> grad = alpha
    EXPECT_FLOAT_EQ(grads[0][{2}], 1.0F);   // x > 0 -> grad = 1
}
```

#### 3.5.4 数値勾配チェック

```cpp
TEST_F(ActivationOpsTest, LeakyReLUNumericalGradient) {
    float alpha = 0.01F;
    auto x = Tensor<float>({-1.0F, 0.5F, 1.0F});
    auto op = std::make_shared<LeakyReLUOperation<float>>(alpha);

    bool grad_correct = ops::test::checkNumericalGradient(
        *op, {x}, {}, 1e-4F, 1e-2F);
    EXPECT_TRUE(grad_correct);
}
```

### 3.6 Softmax

#### 3.6.1 Forward テスト（基本）

**目的**: Softmax の出力が確率分布（総和 = 1）になることを確認

```cpp
TEST_F(ActivationOpsTest, SoftmaxForward) {
    auto x = Tensor<float>({1.0F, 2.0F, 3.0F});
    auto op = std::make_shared<SoftmaxOperation<float>>();
    auto result = op->forward({x});

    EXPECT_EQ(result.shape(), Shape({3}));

    // 総和が 1 であることを確認
    float sum = result[{0}] + result[{1}] + result[{2}];
    EXPECT_NEAR(sum, 1.0F, 1e-5F);

    // すべての要素が [0, 1] の範囲内
    EXPECT_GT(result[{0}], 0.0F);
    EXPECT_LT(result[{0}], 1.0F);
    EXPECT_GT(result[{1}], 0.0F);
    EXPECT_LT(result[{1}], 1.0F);
    EXPECT_GT(result[{2}], 0.0F);
    EXPECT_LT(result[{2}], 1.0F);

    // 大きい入力ほど大きい出力
    EXPECT_LT(result[{0}], result[{1}]);
    EXPECT_LT(result[{1}], result[{2}]);
}
```

#### 3.6.2 Forward テスト（2D テンソル）

**目的**: 2D テンソルでも dim 指定で正しく動作することを確認

```cpp
TEST_F(ActivationOpsTest, Softmax2D) {
    auto x = Tensor<float>({{1.0F, 2.0F, 3.0F},
                            {4.0F, 5.0F, 6.0F}});
    auto op = std::make_shared<SoftmaxOperation<float>>(-1);  // 最後の次元
    auto result = op->forward({x});

    EXPECT_EQ(result.shape(), Shape({2, 3}));

    // 各行の総和が 1 であることを確認
    float sum_row0 = result[{0, 0}] + result[{0, 1}] + result[{0, 2}];
    float sum_row1 = result[{1, 0}] + result[{1, 1}] + result[{1, 2}];
    EXPECT_NEAR(sum_row0, 1.0F, 1e-5F);
    EXPECT_NEAR(sum_row1, 1.0F, 1e-5F);
}
```

#### 3.6.3 数値安定性テスト（大きな値）

**目的**: 大きな値でオーバーフローしないことを確認

```cpp
TEST_F(ActivationOpsTest, SoftmaxNumericalStabilityLarge) {
    auto x = Tensor<float>({1000.0F, 1001.0F, 1002.0F});
    auto op = std::make_shared<SoftmaxOperation<float>>();
    auto result = op->forward({x});

    // すべての要素が有限値であることを確認
    EXPECT_TRUE(std::isfinite(result[{0}]));
    EXPECT_TRUE(std::isfinite(result[{1}]));
    EXPECT_TRUE(std::isfinite(result[{2}]));

    // 総和が 1 であることを確認
    float sum = result[{0}] + result[{1}] + result[{2}];
    EXPECT_NEAR(sum, 1.0F, 1e-5F);
}
```

#### 3.6.4 数値安定性テスト（小さな値）

**目的**: 小さな値でアンダーフローしないことを確認

```cpp
TEST_F(ActivationOpsTest, SoftmaxNumericalStabilitySmall) {
    auto x = Tensor<float>({-1000.0F, -1001.0F, -1002.0F});
    auto op = std::make_shared<SoftmaxOperation<float>>();
    auto result = op->forward({x});

    // すべての要素が有限値であることを確認
    EXPECT_TRUE(std::isfinite(result[{0}]));
    EXPECT_TRUE(std::isfinite(result[{1}]));
    EXPECT_TRUE(std::isfinite(result[{2}]));

    // 総和が 1 であることを確認
    float sum = result[{0}] + result[{1}] + result[{2}];
    EXPECT_NEAR(sum, 1.0F, 1e-5F);
}
```

#### 3.6.5 Backward テスト

```cpp
TEST_F(ActivationOpsTest, SoftmaxBackward) {
    auto x = Tensor<float>({1.0F, 2.0F, 3.0F});
    auto op = std::make_shared<SoftmaxOperation<float>>();
    auto output = op->forward({x});

    auto grad_output = Tensor<float>({1.0F, 0.0F, 0.0F});
    auto grads = op->backward(grad_output);

    ASSERT_EQ(grads.size(), 1UL);
    EXPECT_EQ(grads[0].shape(), x.shape());

    // 勾配の総和が 0 であることを確認（Softmax の性質）
    float grad_sum = grads[0][{0}] + grads[0][{1}] + grads[0][{2}];
    EXPECT_NEAR(grad_sum, 0.0F, 1e-5F);
}
```

#### 3.6.6 数値勾配チェック

```cpp
TEST_F(ActivationOpsTest, SoftmaxNumericalGradient) {
    auto x = Tensor<float>({1.0F, 2.0F, 3.0F});
    auto op = std::make_shared<SoftmaxOperation<float>>();

    bool grad_correct = ops::test::checkNumericalGradient(
        *op, {x}, {}, 1e-4F, 1e-2F);
    EXPECT_TRUE(grad_correct);
}
```

### 3.7 LogSoftmax

#### 3.7.1 Forward テスト

**目的**: LogSoftmax の出力が log 確率になることを確認

```cpp
TEST_F(ActivationOpsTest, LogSoftmaxForward) {
    auto x = Tensor<float>({1.0F, 2.0F, 3.0F});
    auto op = std::make_shared<LogSoftmaxOperation<float>>();
    auto result = op->forward({x});

    EXPECT_EQ(result.shape(), Shape({3}));

    // すべての要素が負の値（log 確率）
    EXPECT_LT(result[{0}], 0.0F);
    EXPECT_LT(result[{1}], 0.0F);
    EXPECT_LT(result[{2}], 0.0F);

    // exp(log_softmax) = softmax を確認
    auto softmax_op = std::make_shared<SoftmaxOperation<float>>();
    auto softmax_result = softmax_op->forward({x});

    EXPECT_NEAR(std::exp(result[{0}]), softmax_result[{0}], 1e-5F);
    EXPECT_NEAR(std::exp(result[{1}]), softmax_result[{1}], 1e-5F);
    EXPECT_NEAR(std::exp(result[{2}]), softmax_result[{2}], 1e-5F);
}
```

#### 3.7.2 数値安定性テスト

**目的**: LogSoftmax が数値的に安定していることを確認

```cpp
TEST_F(ActivationOpsTest, LogSoftmaxNumericalStability) {
    auto x_large = Tensor<float>({1000.0F, 1001.0F, 1002.0F});
    auto op_large = std::make_shared<LogSoftmaxOperation<float>>();
    auto result_large = op_large->forward({x_large});

    // すべての要素が有限値であることを確認
    EXPECT_TRUE(std::isfinite(result_large[{0}]));
    EXPECT_TRUE(std::isfinite(result_large[{1}]));
    EXPECT_TRUE(std::isfinite(result_large[{2}]));

    auto x_small = Tensor<float>({-1000.0F, -1001.0F, -1002.0F});
    auto op_small = std::make_shared<LogSoftmaxOperation<float>>();
    auto result_small = op_small->forward({x_small});

    EXPECT_TRUE(std::isfinite(result_small[{0}]));
    EXPECT_TRUE(std::isfinite(result_small[{1}]));
    EXPECT_TRUE(std::isfinite(result_small[{2}]));
}
```

#### 3.7.3 Backward テスト

```cpp
TEST_F(ActivationOpsTest, LogSoftmaxBackward) {
    auto x = Tensor<float>({1.0F, 2.0F, 3.0F});
    auto op = std::make_shared<LogSoftmaxOperation<float>>();
    auto output = op->forward({x});

    auto grad_output = Tensor<float>({1.0F, 0.0F, 0.0F});
    auto grads = op->backward(grad_output);

    ASSERT_EQ(grads.size(), 1UL);
    EXPECT_EQ(grads[0].shape(), x.shape());

    // 勾配が有限値であることを確認
    EXPECT_TRUE(std::isfinite(grads[0][{0}]));
    EXPECT_TRUE(std::isfinite(grads[0][{1}]));
    EXPECT_TRUE(std::isfinite(grads[0][{2}]));
}
```

#### 3.7.4 数値勾配チェック

```cpp
TEST_F(ActivationOpsTest, LogSoftmaxNumericalGradient) {
    auto x = Tensor<float>({1.0F, 2.0F, 3.0F});
    auto op = std::make_shared<LogSoftmaxOperation<float>>();

    bool grad_correct = ops::test::checkNumericalGradient(
        *op, {x}, {}, 1e-4F, 1e-2F);
    EXPECT_TRUE(grad_correct);
}
```

## 4. テスト実行計画

### 4.1 Phase 1: 単体テスト（各活性化関数）

**目標**: 各活性化関数の基本機能を検証

1. ReLU の forward/backward テスト
2. Sigmoid の forward/backward テスト
3. Tanh の forward/backward テスト
4. GELU の forward/backward テスト
5. LeakyReLU の forward/backward テスト
6. Softmax の forward/backward テスト
7. LogSoftmax の forward/backward テスト

**完了基準**: すべてのテストが pass

### 4.2 Phase 2: 数値勾配チェック

**目標**: 自動微分の精度を検証

1. 各活性化関数で数値勾配チェックを実行
2. 相対誤差が閾値以下であることを確認

**完了基準**: すべての数値勾配チェックが pass（相対誤差 < 1e-2）

### 4.3 Phase 3: 数値安定性テスト

**目標**: Softmax/LogSoftmax の数値安定性を検証

1. 大きな値（1000.0）でオーバーフローしないことを確認
2. 小さな値（-1000.0）でアンダーフローしないことを確認
3. すべての出力が有限値であることを確認

**完了基準**: すべての数値安定性テストが pass

### 4.4 Phase 4: エッジケーステスト

**目標**: 境界値や特殊ケースでの動作を検証

1. 境界値（0, ±∞）での動作確認
2. 多次元テンソル（2D, 3D）での動作確認
3. パラメータ（LeakyReLU の alpha）の動作確認

**完了基準**: すべてのエッジケーステストが pass

### 4.5 Phase 5: 統合テスト

**目標**: 複数の活性化関数を組み合わせた動作を検証

1. ReLU → Softmax のパイプラインテスト
2. GELU → LayerNorm → Softmax のパイプラインテスト（LayerNorm は将来実装）

**完了基準**: 統合テストが pass

## 5. テストカバレッジ目標

**最低カバレッジ目標**: 90%

**カバーすべき領域**:
- ✅ すべての forward 関数
- ✅ すべての backward 関数
- ✅ すべてのパラメータ化コンストラクタ
- ✅ エッジケース（境界値、極端な値）
- ✅ 数値安定性（オーバーフロー/アンダーフロー）

## 6. テスト実行環境

**プラットフォーム**:
- macOS (Apple Silicon / Intel)
- Linux (Ubuntu 22.04)

**コンパイラ**:
- Clang 15+
- GCC 11+

**CI チェック**:
- ✅ Build & Test (全プラットフォーム)
- ✅ Clang-Tidy
- ✅ Sanitizers (ASAN, UBSAN)
- ✅ Code Format (clang-format)
- ✅ Code Coverage

## 7. 期待される結果

すべてのテストが pass し、以下の基準を満たすことを期待します：

- ✅ すべての forward テストが pass（100%）
- ✅ すべての backward テストが pass（100%）
- ✅ すべての数値勾配チェックが pass（相対誤差 < 1e-2）
- ✅ Softmax/LogSoftmax の数値安定性テストが pass（100%）
- ✅ すべてのエッジケーステストが pass（100%）
- ✅ コードカバレッジ ≥ 90%
- ✅ すべての CI チェックが pass

## 8. トラブルシューティング

### 8.1 数値勾配チェックが失敗する場合

**原因**:
- backward の実装が間違っている
- 数値勾配の epsilon が不適切
- 許容誤差が厳しすぎる

**対策**:
1. backward の数式を再確認
2. epsilon を調整（1e-4 → 1e-3）
3. 許容誤差を調整（1e-2 → 5e-2）
4. 特定の要素でのみ失敗する場合は、その要素を詳細にデバッグ

### 8.2 Softmax でオーバーフローする場合

**原因**:
- log-sum-exp トリックが実装されていない
- 最大値でのシフトが不十分

**対策**:
1. log-sum-exp トリックを正しく実装
2. 最大値を正しく計算してシフト
3. 極端な値（1000.0）でテスト

### 8.3 GELU の数値勾配チェックが失敗する場合

**原因**:
- GELU は tanh 近似版のため、数値誤差が大きい
- 許容誤差が厳しすぎる

**対策**:
1. 許容誤差を 5e-2 に緩める
2. backward の実装を再確認
3. PyTorch の実装と比較

## 9. 参考資料

### テスト手法
- [Google Test Documentation](https://google.github.io/googletest/)
- [Numerical Gradient Checking](https://cs231n.github.io/neural-networks-3/#gradcheck)

### 数値安定性
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)
- [Numerically Stable Softmax](https://jaykmody.com/blog/stable-softmax/)

### PyTorch テスト
- [PyTorch Test Suite](https://github.com/pytorch/pytorch/tree/main/test)
