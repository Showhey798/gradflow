# Issue #13: Phase 2 統合テスト - XOR 問題

## 概要

Phase 2 で実装した自動微分機能（Variable, Operation, Activation, Loss, Optimizer）を統合的にテストします。XOR 問題を 2 層ニューラルネットワークで解くことで、以下の機能が正しく連携することを確認します。

## 目的

1. **自動微分の正確性**: 勾配が正しく計算されること
2. **学習の収束性**: Optimizer により学習が収束すること
3. **統合動作**: すべてのコンポーネントが連携して動作すること
4. **メモリ管理**: メモリリークがないこと

## 依存コンポーネント

すべて実装済み：

- ✅ Variable クラス (PR #59)
- ✅ 基本演算の Operation 実装 (PR #60)
- ✅ 活性化関数 (PR #61)
- ✅ 損失関数 (PR #63)
- ✅ Optimizer (PR #64)

## 実装ファイル

```
tests/test_phase2_integration.cpp
```

## テスト構成

### 1. Phase2IntegrationTest クラス

Phase 1 の統合テストと同様に、GTest のフィクスチャクラスを定義します。

```cpp
class Phase2IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override;
    void TearDown() override;

    // Helper functions
    bool approx_equal(float a, float b, float epsilon = 1e-5f);
    size_t getLargeTestSize(size_t normal_size);
    size_t getIterationCount(size_t normal_count);
};
```

### 2. XOR 問題の統合テスト

**テスト名**: `Phase2IntegrationTest.SimpleNeuralNetwork`

#### 目的
- 2 層ニューラルネットワークで XOR 問題を学習
- 勾配計算、backward、Optimizer の連携を確認
- 学習が収束し、高精度で XOR を解けることを確認

#### テストの流れ

1. **データ準備**: XOR 問題の入力と正解ラベル
   ```
   X = [[0, 0], [0, 1], [1, 0], [1, 1]]
   y = [[0],    [1],    [1],    [0]]
   ```

2. **モデル構築**: 2 層ニューラルネットワーク
   - 入力層: 2 次元
   - 隠れ層: 4 次元（ReLU 活性化）
   - 出力層: 1 次元（Sigmoid 活性化）

3. **パラメータ初期化**:
   - W1: [2, 4] - Xavier/He 初期化（randn）
   - b1: [1, 4] - ゼロ初期化（broadcasting 互換性のため）
   - W2: [4, 1] - Xavier/He 初期化（randn）
   - b2: [1, 1] - ゼロ初期化（broadcasting 互換性のため）

4. **Optimizer**: Adam (lr=0.1) ※高速収束のため学習率を 0.01 から 0.1 に変更

5. **学習ループ**:
   ```cpp
   for (int epoch = 0; epoch < 10000; ++epoch) {  // ※イテレーション数を 5000 から 10000 に変更
       // Forward pass
       auto X_var = Variable<float>(X, false);
       auto h = relu(matmul(X_var, W1) + b1);
       auto y_pred = sigmoid(matmul(h, W2) + b2);
       auto loss = mse_loss(y_pred, Variable<float>(y, false));

       // Backward pass
       optimizer.zeroGrad();
       loss.backward();
       optimizer.step();
   }
   ```

6. **精度検証**:
   - 学習後の予測値を確認
   - 最終ロスが 0.1 未満であることを確認
   - 各サンプルの予測精度を個別に検証
     - XOR(0, 0) = 0: 予測値 < 0.2
     - XOR(0, 1) = 1: 予測値 > 0.8
     - XOR(1, 0) = 1: 予測値 > 0.8
     - XOR(1, 1) = 0: 予測値 < 0.2
   - 正解率 100% を期待

#### 完了基準

- ✅ XOR が正しく解ける（精度 > 90%）
- ✅ 学習ループがクラッシュしない
- ✅ 勾配が正しく計算される（backward が動作）
- ✅ メモリリークゼロ（AddressSanitizer でチェック）

### 3. 勾配計算の正確性テスト（オプション）

**テスト名**: `Phase2IntegrationTest.GradientAccuracy`

#### 目的
- 簡単な関数で、自動微分による勾配と数値勾配を比較
- backward の正確性を数値的に検証

#### テストの流れ

1. 簡単な計算グラフを構築:
   ```cpp
   auto x = Variable<float>(Tensor<float>({2.0, 3.0}), true);
   auto y = Variable<float>(Tensor<float>({1.0, -1.0}), true);
   auto z = sum((x * y) + x);  // z = (2*1 + 2) + (3*(-1) + 3) = 4 + 0 = 4
   ```

2. backward を実行:
   ```cpp
   z.backward();
   ```

3. 勾配を確認:
   ```cpp
   // dz/dx = y + 1 = [1+1, -1+1] = [2, 0]
   EXPECT_FLOAT_EQ(x.grad()[{0}], 2.0f);
   EXPECT_FLOAT_EQ(x.grad()[{1}], 0.0f);

   // dz/dy = x = [2, 3]
   EXPECT_FLOAT_EQ(y.grad()[{0}], 2.0f);
   EXPECT_FLOAT_EQ(y.grad()[{1}], 3.0f);
   ```

### 4. メモリ管理テスト

**テスト名**: `Phase2IntegrationTest.MemoryManagement`

#### 目的
- 複数回の forward/backward でメモリリークがないことを確認
- AddressSanitizer で検出

#### テストの流れ

1. 小規模なニューラルネットワークで 100 回の学習を実行
2. 各イテレーションで Variable が適切に破棄されることを確認
3. Sanitizer 実行時は反復回数を減らす（タイムアウト対策）

```cpp
const size_t kNumIterations = getIterationCount(100);

for (size_t i = 0; i < kNumIterations; ++i) {
    auto X = Variable<float>(Tensor<float>({1.0, 2.0}), false);
    auto W = Variable<float>(Tensor<float>::randn({2, 1}), true);
    auto y_pred = sigmoid(matmul(X, W));
    auto y_true = Variable<float>(Tensor<float>({0.5}), false);
    auto loss = mse_loss(y_pred, y_true);

    loss.backward();
    // Variables should be properly deallocated
}
```

### 5. 複雑な計算グラフテスト（オプション）

**テスト名**: `Phase2IntegrationTest.ComplexComputationGraph`

#### 目的
- より深いネットワークや複雑な計算グラフでも動作することを確認

#### テストの内容

- 3〜4 層のニューラルネットワーク
- 複数の活性化関数の組み合わせ
- Batch Normalization 的な正規化（将来的に実装された場合）

## 実装上の注意点

### 1. Sanitizer 対応

Phase 1 と同様に、AddressSanitizer/ThreadSanitizer 実行時は処理を軽量化：

```cpp
size_t getLargeTestSize(size_t normal_size) {
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
    return std::max(size_t{10}, normal_size / 50);
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    return std::max(size_t{10}, normal_size / 50);
#endif
    return normal_size;
}
```

### 2. 学習の収束性

XOR 問題は非線形なので、学習が収束しない可能性があります。以下の対策：

1. **十分な Epoch 数**: 10000 epoch（Sanitizer 実行時は自動的に 1/20 に削減）
2. **適切な学習率**: Adam で lr=0.1（高速収束のため）
3. **隠れ層のサイズ**: 4 次元で十分
4. **初期化**: Xavier/He 初期化（randn で近似）
5. **乱数シード固定**: `Tensor<float>::setSeed(42)` で再現性を確保

### 3. テストの安定性

乱数初期化による学習の不安定性を防ぐ：

1. **シード固定**: SetUp() で `Tensor<float>::setSeed(42)` を呼び出して再現性を確保
2. **厳格な閾値**: 最終ロス < 0.1、各サンプルの予測精度 > 80% を要求
3. **正解率検証**: 全サンプルで正解率 100% を期待

### 4. 必要なインクルード

```cpp
#include <gtest/gtest.h>

#include "gradflow/autograd/variable.hpp"
#include "gradflow/autograd/ops/matmul.hpp"
#include "gradflow/autograd/ops/relu.hpp"
#include "gradflow/autograd/ops/sigmoid.hpp"
#include "gradflow/autograd/ops/loss.hpp"
#include "gradflow/optim/adam.hpp"
```

## 期待される成果物

1. **テストファイル**: `tests/test_phase2_integration.cpp`
   - 最低限: `SimpleNeuralNetwork` テスト
   - オプション: `GradientAccuracy`, `MemoryManagement`, `ComplexComputationGraph` テスト

2. **テスト実行**:
   ```bash
   # ビルド
   cmake --build build --target test_phase2_integration

   # 実行
   ./build/tests/test_phase2_integration

   # Sanitizer 付き実行
   cmake -DCMAKE_BUILD_TYPE=Debug \
         -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer" \
         -S . -B build-sanitizer
   cmake --build build-sanitizer --target test_phase2_integration
   ./build-sanitizer/tests/test_phase2_integration
   ```

3. **ドキュメント更新**:
   - PROGRESS.md に Phase 2 完了を記録
   - ROADMAP.md の Phase 2 を完了としてマーク

## 完了条件

- ✅ `test_phase2_integration.cpp` が実装されている
- ✅ `SimpleNeuralNetwork` テストが PASS
- ✅ XOR 問題が 90% 以上の精度で解ける
- ✅ AddressSanitizer でメモリリークゼロ
- ✅ すべての CI チェックが PASS

## 参考資料

- Phase 1 統合テスト: `tests/test_phase1_integration.cpp`
- PyTorch のチュートリアル: XOR 問題の学習例
- 自動微分の正確性: 数値勾配チェックの実装パターン
