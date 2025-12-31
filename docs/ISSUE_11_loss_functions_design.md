# Issue #11: 損失関数の実装 - 設計書

## 概要

機械学習の損失関数（Loss Functions）とその勾配を自動微分機能付きで実装します。
Phase 2.5 の一環として、ニューラルネットワークの学習に必要な 4 つの主要な損失関数を実装します。

## 1. 調査・リサーチ結果

### 既存ライブラリの損失関数実装

#### PyTorch
- `torch.nn.MSELoss`: 回帰タスク向け、reduction パラメータで平均/合計を選択
- `torch.nn.CrossEntropyLoss`: LogSoftmax + NLLLoss の組み合わせ、数値安定性を考慮
- `torch.nn.BCELoss`: 二値分類向け、入力は [0, 1] 範囲（sigmoid 適用済み）
- `torch.nn.NLLLoss`: 負の対数尤度、クラス分類の基本損失

#### TensorFlow/Keras
- `tf.keras.losses.MeanSquaredError`: PyTorch と同様
- `tf.keras.losses.CategoricalCrossentropy`: from_logits パラメータで数値安定版を提供
- `tf.keras.losses.BinaryCrossentropy`: BCE の標準実装

#### JAX
- `optax.l2_loss`: MSE の基礎
- `optax.softmax_cross_entropy`: 数値安定な実装（log-sum-exp トリック）

### 参考文献
- [PyTorch Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions) - API 設計の参考
- [TensorFlow Losses](https://www.tensorflow.org/api_docs/python/tf/keras/losses) - 数値安定性の実装パターン
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/neural-networks-2/) - 損失関数の理論背景

---

## 2. 分析と評価

### 現状の課題
- 損失関数が未実装のため、ニューラルネットワークの学習ができない
- 自動微分機能は整っているが、損失計算とその勾配が必要

### 採用すべき設計原則
- **単一責任の原則**: 各損失関数は 1 つの損失計算のみを担当
- **数値安定性**: オーバーフロー/アンダーフローを防ぐ実装（log-sum-exp トリック等）
- **一貫性**: 既存の Operation クラス（ReLU, Sigmoid 等）と同じインターフェース
- **テスト駆動**: 数値勾配チェックで backward の正確性を検証

---

## 3. 推奨アーキテクチャ案

### 設計のコンセプト

すべての損失関数は `Operation<T>` 基底クラスを継承し、`forward()` と `backward()` を実装します。
数値安定性を最優先し、既存の Softmax 実装（log-sum-exp トリック）と同様のアプローチを採用します。

### 実装する損失関数

#### 3.1 MSELossOperation (Mean Squared Error)

**用途**: 回帰タスク

**数学的定義**:
```
Forward:
  L = (1/N) * Σ(predicted - target)²

Backward:
  ∂L/∂predicted = (2/N) * (predicted - target)
```

**実装のポイント**:
- 要素ごとの差の二乗を計算
- 平均を取る（reduction=mean）
- シンプルで数値的に安定

**C++ 実装イメージ**:
```cpp
template <typename T>
class MSELossOperation : public Operation<T> {
public:
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        // inputs[0]: predicted, inputs[1]: target
        auto diff = sub(inputs[0], inputs[1]);
        auto squared = mul(diff, diff);
        auto sum_loss = sum(squared);  // scalar
        // mean: divide by total elements
        Tensor<T> loss(Shape{});
        loss.data()[0] = sum_loss.data()[0] / inputs[0].size();

        this->saveForBackward("predicted", inputs[0]);
        this->saveForBackward("target", inputs[1]);
        return loss;
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto predicted = this->getSavedTensor("predicted");
        auto target = this->getSavedTensor("target");

        // grad = (2/N) * (predicted - target) * grad_output
        auto diff = sub(predicted, target);
        T scale = T(2) / static_cast<T>(predicted.size());

        Tensor<T> scaled_diff(diff.shape());
        for (size_t i = 0; i < diff.size(); ++i) {
            scaled_diff.data()[i] = scale * diff.data()[i] * grad_output.data()[0];
        }

        return {scaled_diff, negate(scaled_diff)};
    }
};
```

---

#### 3.2 CrossEntropyLossOperation

**用途**: 多クラス分類タスク

**数学的定義**:
```
Forward:
  log_probs = log_softmax(logits)  # numerically stable
  L = -Σ(target * log_probs)

Backward:
  ∂L/∂logits = softmax(logits) - target
```

**実装のポイント**:
- **数値安定性が最重要**: logits が大きいとき softmax で overflow する
- log-sum-exp トリック: `log(softmax(x)) = x - log(sum(exp(x)))`
- さらに安定化: `log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))`
- target は one-hot エンコード形式

**C++ 実装イメージ**:
```cpp
template <typename T>
class CrossEntropyLossOperation : public Operation<T> {
public:
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        // inputs[0]: logits [N, C], inputs[1]: target [N, C] (one-hot)
        const auto& logits = inputs[0];
        const auto& target = inputs[1];

        // log_softmax(logits): numerically stable
        // log_softmax(x) = x - log(sum(exp(x)))
        //                = x - (max(x) + log(sum(exp(x - max(x)))))

        auto max_logits = max(logits, 1, /*keepdim=*/true);
        auto shifted = sub(logits, max_logits);
        auto exp_shifted = exp(shifted);
        auto sum_exp = sum(exp_shifted, 1, /*keepdim=*/true);
        auto log_sum_exp = log(sum_exp);
        auto log_sum_exp_shifted = add(max_logits, log_sum_exp);
        auto log_probs = sub(logits, log_sum_exp_shifted);

        // -Σ(target * log_probs)
        auto target_log_probs = mul(target, log_probs);
        auto sum_loss = sum(target_log_probs);

        Tensor<T> loss(Shape{});
        loss.data()[0] = -sum_loss.data()[0] / target.shape()[0];  // mean over batch

        this->saveForBackward("logits", logits);
        this->saveForBackward("target", target);
        return loss;
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto logits = this->getSavedTensor("logits");
        auto target = this->getSavedTensor("target");

        // grad = (softmax(logits) - target) / N
        // Compute softmax(logits)
        auto max_logits = max(logits, 1, /*keepdim=*/true);
        auto shifted = sub(logits, max_logits);
        auto exp_shifted = exp(shifted);
        auto sum_exp = sum(exp_shifted, 1, /*keepdim=*/true);
        auto probs = div(exp_shifted, sum_exp);

        auto grad_logits = sub(probs, target);

        // Scale by grad_output and batch size
        T scale = grad_output.data()[0] / static_cast<T>(logits.shape()[0]);
        for (size_t i = 0; i < grad_logits.size(); ++i) {
            grad_logits.data()[i] *= scale;
        }

        return {grad_logits, Tensor<T>(target.shape())};  // target has no gradient
    }
};
```

---

#### 3.3 BCELossOperation (Binary Cross Entropy)

**用途**: 二値分類タスク

**数学的定義**:
```
Forward:
  L = -(1/N) * Σ[target * log(predicted) + (1-target) * log(1-predicted)]

Backward:
  ∂L/∂predicted = -(1/N) * [target/predicted - (1-target)/(1-predicted)]
```

**実装のポイント**:
- 入力 `predicted` は [0, 1] 範囲（sigmoid 適用済み）を想定
- 数値安定性: `log(0)` を防ぐため小さなイプシロンを追加: `log(predicted + ε)`
- PyTorch では `BCEWithLogitsLoss`（logits から直接計算）も提供されているが、今回はシンプルな BCE

**C++ 実装イメージ**:
```cpp
template <typename T>
class BCELossOperation : public Operation<T> {
public:
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        // inputs[0]: predicted [N], inputs[1]: target [N]
        const auto& predicted = inputs[0];
        const auto& target = inputs[1];

        // Numerical stability: clamp predicted to [eps, 1-eps]
        const T eps = static_cast<T>(1e-7);
        Tensor<T> clamped_pred(predicted.shape());
        for (size_t i = 0; i < predicted.size(); ++i) {
            clamped_pred.data()[i] = std::max(eps, std::min(T(1) - eps, predicted.data()[i]));
        }

        // BCE: -[target * log(pred) + (1-target) * log(1-pred)]
        Tensor<T> loss_elements(predicted.shape());
        for (size_t i = 0; i < predicted.size(); ++i) {
            T pred_val = clamped_pred.data()[i];
            T target_val = target.data()[i];
            loss_elements.data()[i] = -(target_val * std::log(pred_val) +
                                        (T(1) - target_val) * std::log(T(1) - pred_val));
        }

        auto sum_loss = sum(loss_elements);
        Tensor<T> loss(Shape{});
        loss.data()[0] = sum_loss.data()[0] / predicted.size();

        this->saveForBackward("predicted", clamped_pred);
        this->saveForBackward("target", target);
        return loss;
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto predicted = this->getSavedTensor("predicted");
        auto target = this->getSavedTensor("target");

        // grad = -(1/N) * [target/pred - (1-target)/(1-pred)]
        Tensor<T> grad_predicted(predicted.shape());
        T scale = grad_output.data()[0] / static_cast<T>(predicted.size());

        for (size_t i = 0; i < predicted.size(); ++i) {
            T pred_val = predicted.data()[i];
            T target_val = target.data()[i];
            grad_predicted.data()[i] = -scale * (target_val / pred_val -
                                                  (T(1) - target_val) / (T(1) - pred_val));
        }

        return {grad_predicted, Tensor<T>(target.shape())};
    }
};
```

---

#### 3.4 NLLLossOperation (Negative Log Likelihood)

**用途**: 多クラス分類（log_softmax の出力と組み合わせ）

**数学的定義**:
```
Forward:
  L = -(1/N) * Σ log_probs[target_class]

Backward:
  ∂L/∂log_probs[i] = -(1/N) if i == target_class else 0
```

**実装のポイント**:
- 入力は log-probabilities（log_softmax の出力）
- target はクラスインデックス（整数）または one-hot
- CrossEntropyLoss は内部で LogSoftmax + NLLLoss を実行
- シンプルで数値的に安定

**C++ 実装イメージ**:
```cpp
template <typename T>
class NLLLossOperation : public Operation<T> {
public:
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        // inputs[0]: log_probs [N, C], inputs[1]: target [N, C] (one-hot)
        const auto& log_probs = inputs[0];
        const auto& target = inputs[1];

        // -Σ log_probs[target_class]
        auto target_log_probs = mul(log_probs, target);
        auto sum_loss = sum(target_log_probs);

        Tensor<T> loss(Shape{});
        loss.data()[0] = -sum_loss.data()[0] / log_probs.shape()[0];

        this->saveForBackward("log_probs", log_probs);
        this->saveForBackward("target", target);
        return loss;
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto log_probs = this->getSavedTensor("log_probs");
        auto target = this->getSavedTensor("target");

        // grad = -target / N
        Tensor<T> grad_log_probs(log_probs.shape());
        T scale = -grad_output.data()[0] / static_cast<T>(log_probs.shape()[0]);

        for (size_t i = 0; i < target.size(); ++i) {
            grad_log_probs.data()[i] = scale * target.data()[i];
        }

        return {grad_log_probs, Tensor<T>(target.shape())};
    }
};
```

---

## 4. テスト戦略

### 4.1 数値勾配チェック（Gradient Check）

すべての損失関数に対して数値勾配チェックを実装します。

**アプローチ**:
```cpp
// Numerical gradient: (f(x + h) - f(x - h)) / (2h)
T numerical_grad(Operation<T>& op, Tensor<T> input, size_t idx, T h = 1e-5) {
    auto original_val = input.data()[idx];

    input.data()[idx] = original_val + h;
    auto loss_plus = op.forward({input, target});

    input.data()[idx] = original_val - h;
    auto loss_minus = op.forward({input, target});

    input.data()[idx] = original_val;

    return (loss_plus.data()[0] - loss_minus.data()[0]) / (T(2) * h);
}

// Compare with analytical gradient
TEST(LossGradTest, MSELossGradient) {
    auto predicted = Tensor<float>::randn({3, 4});
    auto target = Tensor<float>::randn({3, 4});

    MSELossOperation<float> op;
    auto loss = op.forward({predicted, target});

    Tensor<float> grad_output({});
    grad_output.data()[0] = 1.0f;
    auto grads = op.backward(grad_output);

    // Check numerical gradient for each element
    for (size_t i = 0; i < predicted.size(); ++i) {
        float numerical = numerical_grad(op, predicted, i);
        float analytical = grads[0].data()[i];
        EXPECT_NEAR(numerical, analytical, 1e-4);
    }
}
```

### 4.2 収束テスト

簡単な最適化問題で損失が減少することを確認します。

**テスト例**:
```cpp
TEST(LossConvergenceTest, MSELossConvergence) {
    // Simple regression: predict constant target
    auto target = Tensor<float>::ones({10, 1});
    auto predicted = Tensor<float>::randn({10, 1});

    MSELossOperation<float> op;

    float learning_rate = 0.1f;
    float prev_loss = std::numeric_limits<float>::max();

    for (int iter = 0; iter < 100; ++iter) {
        auto loss = op.forward({predicted, target});
        auto grads = op.backward(Tensor<float>({}, 1.0f));

        // Gradient descent step
        for (size_t i = 0; i < predicted.size(); ++i) {
            predicted.data()[i] -= learning_rate * grads[0].data()[i];
        }

        // Loss should decrease
        EXPECT_LT(loss.data()[0], prev_loss + 1e-5);
        prev_loss = loss.data()[0];
    }

    // Final loss should be close to 0
    EXPECT_LT(prev_loss, 0.01f);
}
```

### 4.3 数値安定性テスト

極端な値（大きな logits、0 に近い確率等）での動作を確認します。

```cpp
TEST(LossStabilityTest, CrossEntropyLargeLogits) {
    // Very large logits should not cause overflow
    Tensor<float> logits({2, 3});
    logits.data()[0] = 1000.0f;  // extreme value
    logits.data()[1] = 999.0f;
    logits.data()[2] = 998.0f;
    // ... (rest of data)

    auto target = Tensor<float>::zeros({2, 3});
    target.data()[0] = 1.0f;  // one-hot

    CrossEntropyLossOperation<float> op;
    auto loss = op.forward({logits, target});

    // Should not be inf or nan
    EXPECT_TRUE(std::isfinite(loss.data()[0]));
}
```

---

## 5. 実装の優先順位

1. **MSELossOperation** (最優先)
   - 最もシンプル
   - 回帰タスクで即座に使用可能
   - 数値勾配チェックの基本パターンを確立

2. **NLLLossOperation**
   - CrossEntropyLoss の基礎
   - シンプルで安定

3. **CrossEntropyLossOperation**
   - 最も複雑だが最重要
   - 数値安定性の実装パターンを確立

4. **BCELossOperation**
   - 二値分類向け
   - 数値安定性への配慮が必要

---

## 6. トレードオフと設計判断

### メリット
- **一貫性**: 既存の Operation と同じインターフェース
- **数値安定性**: log-sum-exp トリック等で overflow/underflow を防止
- **テスト駆動**: 数値勾配チェックで backward の正確性を保証
- **拡張性**: 新しい損失関数を同じパターンで追加可能

### リスク/注意点
- **パフォーマンス**: 数値安定性のために追加計算が必要（log-sum-exp）
  - 対策: 将来的に SIMD や GPU での最適化を検討
- **メモリ使用**: 中間テンソルの保存
  - 対策: 必要最小限のテンソルのみ保存
- **精度**: float vs double のトレードオフ
  - 対策: テンプレートで両方サポート、テストで許容誤差を調整

---

## 7. ソフトウェア工学的な観点

### 適用するデザインパターン
- **Template Method**: Operation 基底クラスが forward/backward のフレームワークを提供
- **Strategy Pattern**: 各損失関数は異なる戦略（MSE, CrossEntropy 等）を実装
- **Composite Pattern**: Tensor 演算を組み合わせて複雑な損失を構築

### SOLID 原則の適用
- **単一責任**: 各損失関数クラスは 1 つの損失計算のみ
- **開放閉鎖**: 新しい損失関数を追加してもベースクラスは変更不要
- **リスコフ置換**: すべての損失関数は Operation として扱える
- **インターフェース分離**: Operation インターフェースは最小限
- **依存性逆転**: 高レベル（Variable, backward）は抽象（Operation）に依存

---

## 8. 実装スケジュール

### Phase 1: 基礎実装（推定: 2-3 日）
- MSELossOperation 実装
- 基本的な数値勾配チェック
- 収束テスト

### Phase 2: 分類損失（推定: 3-4 日）
- NLLLossOperation 実装
- CrossEntropyLossOperation 実装（数値安定版）
- 数値安定性テスト

### Phase 3: 二値分類（推定: 1-2 日）
- BCELossOperation 実装
- 包括的なテストスイート完成

### Phase 4: ドキュメントと統合（推定: 1 日）
- README への使用例追加
- Phase 2.5 統合テスト
- コードレビューと最終調整

**総推定期間**: 7-10 日

---

## 9. ファイル構成

```
include/gradflow/autograd/ops/
└── loss.hpp                    # すべての損失関数を含む単一ヘッダー

tests/
└── test_loss_grad.cpp         # 損失関数のテスト

docs/
└── ISSUE_11_loss_functions_design.md  # 本設計書
```

---

## 10. 完了基準

- [ ] MSELossOperation の実装と数値勾配チェック
- [ ] CrossEntropyLossOperation の実装と数値勾配チェック
- [ ] BCELossOperation の実装と数値勾配チェック
- [ ] NLLLossOperation の実装と数値勾配チェック
- [ ] すべての数値勾配チェックが pass（相対誤差 < 1e-3）
- [ ] 収束テストがすべて pass
- [ ] 数値安定性テストが pass（inf/nan なし）
- [ ] CI チェックがすべて pass
- [ ] コードレビュー完了

---

## まとめ

本設計書では、機械学習の 4 つの主要な損失関数（MSE, CrossEntropy, BCE, NLL）の実装方針を定めました。
既存の Operation パターンを踏襲しつつ、数値安定性を最優先とした実装を行います。
数値勾配チェックと収束テストにより、backward の正確性と実用性を保証します。
