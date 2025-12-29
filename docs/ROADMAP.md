# Implementation Roadmap

## 目次
- [概要](#概要)
- [フェーズ構成](#フェーズ構成)
- [Phase 1: 基礎インフラ](#phase-1-基礎インフラ)
- [Phase 2: 自動微分の基本機能](#phase-2-自動微分の基本機能)
- [Phase 3: GPU サポート](#phase-3-gpu-サポート)
- [Phase 4: 高度な演算と最適化](#phase-4-高度な演算と最適化)
- [Phase 5: Transformer コンポーネント](#phase-5-transformer-コンポーネント)
- [Phase 6: 統合とベンチマーク](#phase-6-統合とベンチマーク)
- [継続的な改善](#継続的な改善)

---

## 概要

本ロードマップは、自動微分ライブラリと Transformer 実装を段階的に構築するための計画です。各フェーズは独立してテスト可能であり、前フェーズの成果物に依存します。

### 開発原則

1. **TDD (Test-Driven Development)**: すべての機能は、テストを先に書いてから実装する
2. **段階的な統合**: 各フェーズの最後に統合テストを実施
3. **パフォーマンス測定**: 各フェーズでベンチマークを実行し、パフォーマンス回帰を防ぐ
4. **ドキュメント優先**: コードと同時にドキュメントを更新
5. **コードレビュー**: すべての変更は PR ベースでレビュー

### 推定期間

- **Phase 1-2**: 4-6 週間（基礎インフラ、自動微分）
- **Phase 3**: 3-4 週間（Metal サポート）
- **Phase 4**: 3-4 週間（高度な演算と最適化）
- **Phase 5**: 4-5 週間（Transformer コンポーネント）
- **Phase 6**: 2-3 週間（統合とベンチマーク）
- **Phase 7-8**: 3-4 週間（静的グラフ、CUDA サポート - オプション）

**合計**: 約 5-7 ヶ月

---

## フェーズ構成

```
Phase 1: 基礎インフラ（動的グラフ）
    ↓
Phase 2: 自動微分の基本機能（動的グラフ）
    ↓
Phase 3: Metal GPU サポート
    ↓
Phase 4: 高度な演算と最適化
    ↓
Phase 5: Transformer コンポーネント
    ↓
Phase 6: 統合とベンチマーク
    ↓
Phase 7: 静的計算グラフの実装（オプション）
    ↓
Phase 8: CUDA サポート（オプション）
```

---

## Phase 1: 基礎インフラ

### 目標

Tensor の基本機能とデバイス抽象化を実装し、CPU での基本演算が動作する状態を構築する。

### 期間: 4-6 週間

### 実装項目

#### 1.1 Shape と Stride (Week 1)

**実装内容**:
- `Shape` クラス: 多次元配列の形状を表現
- `Stride` クラス: メモリレイアウトを管理
- Broadcasting のルール実装

**ファイル**:
- `include/gradflow/autograd/shape.hpp`
- `tests/test_shape.cpp`

**テスト項目**:
```cpp
TEST(ShapeTest, Construction) { ... }
TEST(ShapeTest, BroadcastCompatibility) { ... }
TEST(StrideTest, RowMajorStride) { ... }
TEST(StrideTest, OffsetCalculation) { ... }
```

**完了基準**:
- すべてのテストがパス
- ドキュメントが完備
- コードカバレッジ > 90%

---

#### 1.2 Storage と DeviceAllocator (Week 1-2)

**実装内容**:
- `Storage<T>` クラス: 実際のメモリバッファを管理
- `DeviceAllocator` 抽象クラス: デバイス固有のメモリ割り当て
- `CPUAllocator` 実装: CPU メモリの割り当て

**ファイル**:
- `include/gradflow/autograd/storage.hpp`
- `include/gradflow/autograd/device.hpp`
- `include/gradflow/autograd/allocator.hpp`
- `tests/test_storage.cpp`

**テスト項目**:
```cpp
TEST(StorageTest, Allocation) { ... }
TEST(StorageTest, DataAccess) { ... }
TEST(CPUAllocatorTest, AlignedAllocation) { ... }
```

**完了基準**:
- メモリリークがないことを Valgrind で確認
- RAII による適切なリソース管理
- コードカバレッジ > 90%

---

#### 1.3 Tensor クラス (Week 2-3)

**実装内容**:
- `Tensor<T>` クラスの基本機能
  - コンストラクタ（形状、データ、初期化リスト）
  - 要素アクセス（operator[]）
  - 形状変換（reshape, view, transpose, permute）
  - スライシング（slice）
  - 連続性チェック（is_contiguous, contiguous）
- ファクトリ関数（zeros, ones, randn, rand, eye）

**ファイル**:
- `include/gradflow/autograd/tensor.hpp`
- `tests/test_tensor.cpp`

**テスト項目**:
```cpp
TEST(TensorTest, Construction) { ... }
TEST(TensorTest, ElementAccess) { ... }
TEST(TensorTest, Reshape) { ... }
TEST(TensorTest, Transpose) { ... }
TEST(TensorTest, Slicing) { ... }
TEST(TensorTest, ViewZeroCopy) { ... }
TEST(TensorTest, Contiguous) { ... }
```

**完了基準**:
- すべての基本操作が動作
- ゼロコピーのビュー操作が正常に動作
- コードカバレッジ > 95%

---

#### 1.4 基本的な CPU 演算 (Week 3-4)

**実装内容**:
- 要素ごとの演算（add, sub, mul, div）
- 行列演算（matmul）
- 集約演算（sum, mean, max, min）
- 数学関数（exp, log, sqrt, pow）

**ファイル**:
- `include/gradflow/autograd/ops/elementwise.hpp`
- `include/gradflow/autograd/ops/matmul.hpp`
- `include/gradflow/autograd/ops/reduction.hpp`
- `tests/test_ops.cpp`

**テスト項目**:
```cpp
TEST(OpsTest, Add) { ... }
TEST(OpsTest, MatMul) { ... }
TEST(OpsTest, Broadcasting) { ... }
TEST(OpsTest, Sum) { ... }
```

**パフォーマンス目標**:
- MatMul: Naive 実装で Eigen の 50% 程度の速度

**完了基準**:
- 数値的正確性を NumPy と比較
- Broadcasting が正しく動作
- コードカバレッジ > 90%

---

#### 1.5 Device 抽象化 (Week 4)

**実装内容**:
- `Device` インターフェース
- `CPUDevice` 実装
- `DeviceManager`: デバイスの取得・管理

**ファイル**:
- `include/gradflow/autograd/device.hpp`
- `src/autograd/device.cpp`
- `tests/test_device.cpp`

**テスト項目**:
```cpp
TEST(DeviceTest, CPUDevice) { ... }
TEST(DeviceManagerTest, GetDevice) { ... }
```

**完了基準**:
- CPU デバイスが正常に動作
- デバイス抽象化が他のコードから使用可能

---

### Phase 1 の統合テスト

```cpp
TEST(Phase1Integration, TensorOperationsOnCPU) {
    auto a = Tensor<float>::randn({100, 200});
    auto b = Tensor<float>::randn({200, 50});

    auto c = matmul(a, b);  // [100, 50]
    auto d = c.transpose(0, 1);  // [50, 100]
    auto e = d.sum(0);  // [100]

    EXPECT_EQ(e.shape(), Shape({100}));
    EXPECT_EQ(e.device(), DeviceType::CPU);
}
```

---

## Phase 2: 自動微分の基本機能

### 目標

計算グラフベースの自動微分機構を実装し、基本的なニューラルネットワークの学習が可能になる。

### 期間: 4-6 週間

### 実装項目

#### 2.1 Operation 基底クラス (Week 1)

**実装内容**:
- `Operation<T>` 抽象クラス
- `forward()` と `backward()` の純粋仮想関数
- 中間変数の保存機構（saved_tensors_）

**ファイル**:
- `include/gradflow/autograd/operation.hpp`
- `tests/test_operation.cpp`

**テスト項目**:
```cpp
TEST(OperationTest, SaveTensors) { ... }
TEST(OperationTest, GetSavedTensors) { ... }
```

**完了基準**:
- 基底クラスが定義され、派生クラスのテンプレートが動作

---

#### 2.2 Variable クラス (Week 1-2)

**実装内容**:
- `Variable<T>` クラス
- `Tensor` のラッパー
- 勾配の保持（grad_）
- 計算グラフへの参照（grad_fn_）
- `backward()` の実装

**ファイル**:
- `include/gradflow/autograd/variable.hpp`
- `tests/test_variable.cpp`

**テスト項目**:
```cpp
TEST(VariableTest, Construction) { ... }
TEST(VariableTest, GradAccumulation) { ... }
TEST(VariableTest, BackwardSimple) { ... }
```

**完了基準**:
- Variable が Tensor をラップして動作
- 勾配が正しく蓄積される

---

#### 2.3 基本演算の Operation 実装 (Week 2-4)

**実装内容**:
- `AddOperation`: 加算とその勾配
- `SubOperation`: 減算とその勾配
- `MulOperation`: 乗算とその勾配
- `DivOperation`: 除算とその勾配
- `MatMulOperation`: 行列積とその勾配
- `PowOperation`: べき乗とその勾配
- `ExpOperation`, `LogOperation`, `SqrtOperation`

**ファイル**:
- `include/gradflow/autograd/ops/add.hpp`
- `include/gradflow/autograd/ops/mul.hpp`
- `include/gradflow/autograd/ops/matmul.hpp`
- ...
- `tests/test_ops_grad.cpp`

**テスト項目**:
```cpp
TEST(OpsGradTest, AddGradient) { ... }
TEST(OpsGradTest, MulGradient) { ... }
TEST(OpsGradTest, MatMulGradient) { ... }
```

**勾配チェック**:
数値微分と自動微分の結果を比較:
```cpp
bool gradient_check(Variable<T>& var, std::function<Variable<T>(Variable<T>&)> func) {
    // 数値微分
    T eps = 1e-5;
    auto numerical_grad = compute_numerical_gradient(var, func, eps);

    // 自動微分
    auto output = func(var);
    output.backward();
    auto auto_grad = var.grad();

    // 比較
    return allclose(numerical_grad, auto_grad, /*rtol=*/1e-3, /*atol=*/1e-5);
}
```

**完了基準**:
- 数値勾配チェックがすべてパス
- コードカバレッジ > 95%

---

#### 2.4 活性化関数 (Week 4-5)

**実装内容**:
- `ReLUOperation`
- `SigmoidOperation`
- `TanhOperation`
- `GELUOperation`
- `LeakyReLUOperation`
- `SoftmaxOperation`
- `LogSoftmaxOperation`

**ファイル**:
- `include/gradflow/autograd/ops/activation.hpp`
- `tests/test_activation_grad.cpp`

**テスト項目**:
```cpp
TEST(ActivationGradTest, ReLUGradient) { ... }
TEST(ActivationGradTest, SigmoidGradient) { ... }
TEST(ActivationGradTest, SoftmaxGradient) { ... }
```

**完了基準**:
- 数値勾配チェックがすべてパス
- Softmax の数値安定性を確認

---

#### 2.5 損失関数 (Week 5)

**実装内容**:
- `MSELossOperation`
- `CrossEntropyLossOperation`
- `BCELossOperation` (Binary Cross Entropy)
- `NLLLossOperation` (Negative Log Likelihood)

**ファイル**:
- `include/gradflow/autograd/ops/loss.hpp`
- `tests/test_loss_grad.cpp`

**テスト項目**:
```cpp
TEST(LossGradTest, MSELossGradient) { ... }
TEST(LossGradTest, CrossEntropyGradient) { ... }
```

**完了基準**:
- 数値勾配チェックがすべてパス
- 損失関数が収束することを簡単な例で確認

---

#### 2.6 Optimizer (Week 5-6)

**実装内容**:
- `Optimizer` 基底クラス
- `SGD`: モーメンタム付き
- `Adam`: adaptive learning rate
- `AdamW`: weight decay 付き Adam

**ファイル**:
- `include/gradflow/optim/optimizer.hpp`
- `include/gradflow/optim/sgd.hpp`
- `include/gradflow/optim/adam.hpp`
- `tests/test_optimizer.cpp`

**テスト項目**:
```cpp
TEST(OptimizerTest, SGDConvergence) { ... }
TEST(OptimizerTest, AdamConvergence) { ... }
```

**完了基準**:
- 簡単な最適化問題（二次関数など）で収束することを確認
- SGD と Adam の両方が動作

---

### Phase 2 の統合テスト

```cpp
TEST(Phase2Integration, SimpleNeuralNetwork) {
    // データ: XOR 問題
    auto X = Tensor<float>({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
    auto y = Tensor<float>({{0}, {1}, {1}, {0}});

    // パラメータ
    auto W1 = Variable<float>::randn({2, 4}, true);
    auto b1 = Variable<float>::zeros({4}, true);
    auto W2 = Variable<float>::randn({4, 1}, true);
    auto b2 = Variable<float>::zeros({1}, true);

    // Optimizer
    std::vector<Variable<float>*> params = {&W1, &b1, &W2, &b2};
    Adam<float> optimizer(params, 0.01);

    // 学習
    for (int epoch = 0; epoch < 5000; ++epoch) {
        // Forward
        auto X_var = Variable<float>(X, false);
        auto h = relu(matmul(X_var, W1) + b1);
        auto y_pred = sigmoid(matmul(h, W2) + b2);

        // Loss
        auto y_var = Variable<float>(y, false);
        auto loss = mse_loss(y_pred, y_var);

        // Backward
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }

    // 最終的な精度を確認
    auto X_var = Variable<float>(X, false);
    auto h = relu(matmul(X_var, W1) + b1);
    auto y_pred = sigmoid(matmul(h, W2) + b2);

    // XOR が解けていることを確認
    EXPECT_LT(std::abs(y_pred.data()[{0, 0}] - 0.0), 0.1);
    EXPECT_GT(y_pred.data()[{1, 0}], 0.9);
    EXPECT_GT(y_pred.data()[{2, 0}], 0.9);
    EXPECT_LT(std::abs(y_pred.data()[{3, 0}] - 0.0), 0.1);
}
```

---

## Phase 3: Metal GPU サポート

### 目標

Metal バックエンドを実装し、Apple Silicon (M1/M2/M3) での GPU 高速計算を実現する。

### 期間: 3-4 週間

### 前提条件

- macOS 12.0 以降
- Xcode Command Line Tools がインストールされている
- Metal Performance Shaders (MPS) が利用可能

### 実装項目

#### 3.1 Metal Device と Allocator (Week 1)

**実装内容**:
- `MetalDevice` 実装
- `MetalAllocator`: GPU メモリの割り当て（MTLBuffer）
- `MTLCommandQueue` のラッパー
- CPU ↔ GPU のデータ転送（Unified Memory の活用）

**ファイル**:
- `include/gradflow/autograd/metal/device.hpp`
- `src/autograd/metal/device.mm`（Objective-C++）
- `tests/test_metal_device.cpp`

**テスト項目**:
```cpp
TEST(MetalDeviceTest, Allocation) { ... }
TEST(MetalDeviceTest, MemoryCopy) { ... }
TEST(MetalDeviceTest, UnifiedMemoryAccess) { ... }
```

**完了基準**:
- GPU メモリの割り当てと解放が動作
- CPU ↔ GPU のデータ転送が動作
- Unified Memory による効率的なメモリアクセス

---

#### 3.2 Metal Compute Shader の実装 (Week 1-2)

**実装内容**:
- 要素ごとの演算（add, mul, etc.）の Metal Compute Shader
- 集約演算（sum, mean）の Metal Compute Shader
- Metal Performance Shaders (MPS) を使った MatMul

**ファイル**:
- `include/gradflow/autograd/metal/kernels.hpp`
- `src/autograd/metal/kernels.metal`（Metal Shading Language）
- `src/autograd/metal/kernels.mm`（C++ ラッパー）
- `tests/test_metal_kernels.cpp`

**Shader の例**:
```metal
kernel void add_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    c[id] = a[id] + b[id];
}
```

**テスト項目**:
```cpp
TEST(MetalKernelsTest, AddKernel) { ... }
TEST(MetalKernelsTest, MatMulMPS) { ... }
```

**パフォーマンス目標**:
- MatMul: MPS の 90% 以上の速度
- Apple Silicon の Neural Engine 活用

**完了基準**:
- CPU と GPU の計算結果が一致（数値誤差を許容）
- GPU が CPU より高速（大きなテンソルで）

---

#### 3.3 MemoryPool の実装 (Week 2-3)

**実装内容**:
- `MemoryPool` クラス
- Best-fit アロケーション
- 断片化の緩和（merge_free_blocks）

**ファイル**:
- `include/gradflow/autograd/memory_pool.hpp`
- `src/autograd/memory_pool.cpp`
- `tests/test_memory_pool.cpp`

**テスト項目**:
```cpp
TEST(MemoryPoolTest, Allocation) { ... }
TEST(MemoryPoolTest, Deallocation) { ... }
TEST(MemoryPoolTest, Fragmentation) { ... }
```

**パフォーマンス測定**:
- メモリプールあり/なしでの割り当て速度を比較

**完了基準**:
- メモリプールが動作
- 断片化が緩和されることを確認

---

#### 3.4 Metal での自動微分 (Week 3-4)

**実装内容**:
- Metal 上での Operation の実装
- 勾配計算の Metal Shader

**ファイル**:
- `src/autograd/metal/ops/*.metal`
- `src/autograd/metal/ops/*.mm`
- `tests/test_metal_ops_grad.cpp`

**テスト項目**:
```cpp
TEST(MetalOpsGradTest, MatMulGradient) { ... }
TEST(MetalOpsGradTest, ReLUGradient) { ... }
```

**完了基準**:
- Metal GPU での勾配計算が CPU と一致
- 数値勾配チェックがすべてパス
- Apple Silicon の GPU を効率的に活用

---

### Phase 3 の統合テスト

```cpp
TEST(Phase3Integration, MetalGPUNeuralNetwork) {
    // データを Metal GPU に配置
    auto X = Variable<float>::randn({1000, 100}, true, DeviceType::Metal);
    auto W1 = Variable<float>::randn({100, 50}, true, DeviceType::Metal);
    auto b1 = Variable<float>::zeros({50}, true, DeviceType::Metal);
    auto W2 = Variable<float>::randn({50, 10}, true, DeviceType::Metal);
    auto b2 = Variable<float>::zeros({10}, true, DeviceType::Metal);

    // Forward
    auto h = relu(matmul(X, W1) + b1);
    auto y = matmul(h, W2) + b2;

    // Backward
    auto loss = y.sum();
    loss.backward();

    // 勾配が計算されていることを確認
    EXPECT_TRUE(W1.grad().device() == DeviceType::Metal);
    EXPECT_GT(W1.grad().abs().sum().data()[{0}], 0.0);

    // CPU に転送して結果を確認（Unified Memory で効率的）
    auto y_cpu = y.to(DeviceType::CPU);
    EXPECT_EQ(y_cpu.shape(), Shape({1000, 10}));
}
```

---

## Phase 4: 高度な演算と最適化

### 目標

パフォーマンスを向上させ、実用的な速度を達成する。

### 期間: 3-4 週間

### 実装項目

#### 4.1 CPU 最適化 (Week 1-2)

**実装内容**:
- SIMD ベクトル化（AVX2/AVX512）
- OpenMP による並列化
- Blocked MatMul（キャッシュ効率を改善）

**ファイル**:
- `src/autograd/cpu/kernels_avx2.cpp`
- `src/autograd/cpu/matmul_blocked.cpp`
- `tests/test_cpu_optimized.cpp`

**パフォーマンス目標**:
- MatMul: Eigen の 80% 以上の速度

**完了基準**:
- ベンチマークで最適化前より高速
- 数値的正確性が保たれている

---

#### 4.2 LayerNorm と Dropout (Week 2)

**実装内容**:
- `LayerNormOperation`
- `DropoutOperation`
- CPU と GPU の両方で実装

**ファイル**:
- `include/gradflow/autograd/ops/layer_norm.hpp`
- `include/gradflow/autograd/ops/dropout.hpp`
- `tests/test_layer_norm_grad.cpp`
- `tests/test_dropout.cpp`

**テスト項目**:
```cpp
TEST(LayerNormGradTest, Gradient) { ... }
TEST(DropoutTest, TrainingMode) { ... }
TEST(DropoutTest, EvalMode) { ... }
```

**完了基準**:
- 数値勾配チェックがパス
- Dropout が train/eval モードで正しく動作

---

#### 4.3 Embedding (Week 2-3)

**実装内容**:
- `EmbeddingOperation`
- 埋め込みの勾配（one-hot → sparse gradient）

**ファイル**:
- `include/gradflow/autograd/ops/embedding.hpp`
- `tests/test_embedding_grad.cpp`

**テスト項目**:
```cpp
TEST(EmbeddingGradTest, Gradient) { ... }
TEST(EmbeddingTest, LookupTable) { ... }
```

**完了基準**:
- Embedding が動作
- 勾配が正しく計算される

---

#### 4.4 カーネル融合 (Week 3-4)

**実装内容**:
- Fused operations（例: Add + ReLU）
- GPU での融合カーネル

**ファイル**:
- `src/autograd/cuda/fused_ops.cu`
- `tests/test_fused_ops.cpp`

**パフォーマンス測定**:
- 融合あり/なしでの実行時間を比較

**完了基準**:
- 融合により速度が向上
- 数値的正確性が保たれている

---

### Phase 4 の統合テスト

```cpp
TEST(Phase4Integration, OptimizedTraining) {
    // 大規模なデータで学習速度を測定
    auto X = Variable<float>::randn({10000, 512}, true, DeviceType::CUDA);
    auto W = Variable<float>::randn({512, 256}, true, DeviceType::CUDA);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; ++i) {
        auto y = matmul(X, W);
        auto loss = y.sum();
        loss.backward();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "100 iterations: " << duration << " ms" << std::endl;

    // 目標: 1 秒以内
    EXPECT_LT(duration, 1000);
}
```

---

## Phase 5: Transformer コンポーネント

### 目標

Transformer の各コンポーネントを実装し、完全な Encoder-Decoder モデルを構築する。

### 期間: 4-5 週間

### 実装項目

#### 5.1 Scaled Dot-Product Attention (Week 1)

**実装内容**:
- `ScaledDotProductAttention` クラス
- Attention weights の計算
- Mask の適用

**ファイル**:
- `include/gradflow/nn/attention.hpp`
- `tests/test_attention.cpp`

**テスト項目**:
```cpp
TEST(AttentionTest, Forward) { ... }
TEST(AttentionTest, Masked) { ... }
TEST(AttentionTest, Gradient) { ... }
```

**完了基準**:
- Attention が動作
- Mask が正しく適用される
- 数値勾配チェックがパス

---

#### 5.2 Multi-Head Attention (Week 1-2)

**実装内容**:
- `MultiHeadAttention` クラス
- Linear projections (Q, K, V, O)
- 複数ヘッドの並列処理

**ファイル**:
- `include/gradflow/nn/multi_head_attention.hpp`
- `tests/test_multi_head_attention.cpp`

**テスト項目**:
```cpp
TEST(MultiHeadAttentionTest, Forward) { ... }
TEST(MultiHeadAttentionTest, Gradient) { ... }
TEST(MultiHeadAttentionTest, SelfAttention) { ... }
TEST(MultiHeadAttentionTest, CrossAttention) { ... }
```

**完了基準**:
- Multi-Head Attention が動作
- Self-attention と Cross-attention が動作
- 数値勾配チェックがパス

---

#### 5.3 Position-wise Feed-Forward Network (Week 2)

**実装内容**:
- `PositionwiseFeedForward` クラス
- 2 層の全結合層

**ファイル**:
- `include/gradflow/nn/feed_forward.hpp`
- `tests/test_feed_forward.cpp`

**テスト項目**:
```cpp
TEST(FeedForwardTest, Forward) { ... }
TEST(FeedForwardTest, Gradient) { ... }
```

**完了基準**:
- FFN が動作
- 数値勾配チェックがパス

---

#### 5.4 Positional Encoding (Week 2)

**実装内容**:
- `PositionalEncoding` クラス
- Sinusoidal positional encoding

**ファイル**:
- `include/gradflow/nn/positional_encoding.hpp`
- `tests/test_positional_encoding.cpp`

**テスト項目**:
```cpp
TEST(PositionalEncodingTest, SinusoidalEncoding) { ... }
TEST(PositionalEncodingTest, Shape) { ... }
```

**完了基準**:
- Positional encoding が生成される
- 形状が正しい

---

#### 5.5 Transformer Encoder Layer (Week 3)

**実装内容**:
- `TransformerEncoderLayer` クラス
- Self-attention + FFN + Residual + LayerNorm

**ファイル**:
- `include/gradflow/nn/transformer_encoder_layer.hpp`
- `tests/test_transformer_encoder_layer.cpp`

**テスト項目**:
```cpp
TEST(EncoderLayerTest, Forward) { ... }
TEST(EncoderLayerTest, Gradient) { ... }
```

**完了基準**:
- Encoder layer が動作
- 数値勾配チェックがパス

---

#### 5.6 Transformer Decoder Layer (Week 3-4)

**実装内容**:
- `TransformerDecoderLayer` クラス
- Masked self-attention + Cross-attention + FFN + Residual + LayerNorm

**ファイル**:
- `include/gradflow/nn/transformer_decoder_layer.hpp`
- `tests/test_transformer_decoder_layer.cpp`

**テスト項目**:
```cpp
TEST(DecoderLayerTest, Forward) { ... }
TEST(DecoderLayerTest, Gradient) { ... }
TEST(DecoderLayerTest, CausalMask) { ... }
```

**完了基準**:
- Decoder layer が動作
- Causal mask が正しく適用される
- 数値勾配チェックがパス

---

#### 5.7 Transformer Encoder (Week 4)

**実装内容**:
- `TransformerEncoder` クラス
- 複数の Encoder layers をスタック

**ファイル**:
- `include/gradflow/nn/transformer_encoder.hpp`
- `tests/test_transformer_encoder.cpp`

**テスト項目**:
```cpp
TEST(EncoderTest, Forward) { ... }
TEST(EncoderTest, Gradient) { ... }
```

**完了基準**:
- Encoder が動作
- 複数レイヤーが正しくスタックされる

---

#### 5.8 Transformer Decoder (Week 4-5)

**実装内容**:
- `TransformerDecoder` クラス
- 複数の Decoder layers をスタック

**ファイル**:
- `include/gradflow/nn/transformer_decoder.hpp`
- `tests/test_transformer_decoder.cpp`

**テスト項目**:
```cpp
TEST(DecoderTest, Forward) { ... }
TEST(DecoderTest, Gradient) { ... }
```

**完了基準**:
- Decoder が動作
- 複数レイヤーが正しくスタックされる

---

#### 5.9 Transformer (Full Model) (Week 5)

**実装内容**:
- `Transformer` クラス
- Encoder + Decoder
- Input/Output embeddings
- Positional encoding

**ファイル**:
- `include/gradflow/nn/transformer.hpp`
- `tests/test_transformer.cpp`

**テスト項目**:
```cpp
TEST(TransformerTest, Forward) { ... }
TEST(TransformerTest, Gradient) { ... }
TEST(TransformerTest, Translation) { ... }
```

**完了基準**:
- 完全な Transformer が動作
- 数値勾配チェックがパス

---

### Phase 5 の統合テスト

```cpp
TEST(Phase5Integration, TransformerTraining) {
    // ダミーの翻訳タスク
    size_t src_vocab = 1000;
    size_t tgt_vocab = 1000;
    size_t d_model = 256;
    size_t num_heads = 8;
    size_t num_layers = 3;

    // モデル
    Transformer<float> model(
        src_vocab, tgt_vocab,
        d_model, num_heads,
        num_layers, num_layers,
        /*d_ff=*/1024, /*dropout=*/0.1
    );

    // データ
    auto src = Tensor<int64_t>::randint(0, src_vocab, {32, 10});  // [batch, src_len]
    auto tgt = Tensor<int64_t>::randint(0, tgt_vocab, {32, 15});  // [batch, tgt_len]

    // Forward
    auto output = model.forward(src, tgt);  // [batch, tgt_len, tgt_vocab]

    // Loss
    auto logits = output.view({-1, tgt_vocab});
    auto labels = tgt.view({-1});
    auto loss = cross_entropy_loss(logits, labels);

    // Backward
    loss.backward();

    // 勾配が計算されていることを確認
    auto params = model.parameters();
    for (auto* param : params) {
        EXPECT_GT(param->grad().abs().sum().data()[{0}], 0.0);
    }
}
```

---

## Phase 6: 統合とベンチマーク

### 目標

実際のタスクで動作することを確認し、パフォーマンスを測定・最適化する。

### 期間: 2-3 週間

### 実装項目

#### 6.1 エンドツーエンドの例 (Week 1)

**実装内容**:
- MNIST 分類の例
- 機械翻訳の例（小規模データセット）
- 言語モデルの例

**ファイル**:
- `examples/mnist_classification.cpp`
- `examples/transformer_translation.cpp`
- `examples/language_model.cpp`

**完了基準**:
- すべての例が動作
- 合理的な精度を達成

---

#### 6.2 ベンチマーク (Week 1-2)

**実装内容**:
- MatMul のベンチマーク（CPU vs GPU, 様々なサイズ）
- Transformer の forward/backward のベンチマーク
- メモリ使用量の測定

**ファイル**:
- `benchmarks/benchmark_matmul.cpp`
- `benchmarks/benchmark_transformer.cpp`
- `benchmarks/benchmark_memory.cpp`

**パフォーマンス目標**:
- CPU MatMul: Eigen の 80% 以上
- GPU MatMul: cuBLAS の 90% 以上
- Transformer forward: PyTorch の 70% 以上（最初の目標）

**完了基準**:
- ベンチマーク結果がドキュメント化される
- ボトルネックが特定される

---

#### 6.3 ドキュメントの完成 (Week 2)

**実装内容**:
- API リファレンスの完成
- チュートリアルの作成
- アーキテクチャ図の作成

**ファイル**:
- `docs/API_REFERENCE.md`
- `docs/TUTORIALS.md`
- `docs/BENCHMARKS.md`

**完了基準**:
- ユーザーがドキュメントを読んで使用できる
- すべての API が文書化されている

---

#### 6.4 Python バインディングの完成 (Week 2-3)

**実装内容**:
- すべての C++ API を Python から使用可能にする
- PyTorch 風の API を提供

**ファイル**:
- `python/bindings.cpp` の拡張
- `python/setup.py` の更新
- Python のテストスイート

**テスト項目**:
```python
def test_transformer_python():
    import gradflow as gf
    import gradflow.nn as nn

    model = nn.Transformer(
        src_vocab=1000, tgt_vocab=1000,
        d_model=256, num_heads=8,
        num_encoder_layers=3, num_decoder_layers=3
    )

    src = gf.randint(0, 1000, (32, 10))
    tgt = gf.randint(0, 1000, (32, 15))

    output = model(src, tgt)
    assert output.shape == (32, 15, 1000)
```

**完了基準**:
- Python から Transformer が使える
- Python のテストがすべてパス

---

#### 6.5 CI/CD の強化 (Week 3)

**実装内容**:
- GPU テストの追加（GitHub Actions + self-hosted runner）
- ベンチマークの自動実行
- パフォーマンス回帰の検出

**ファイル**:
- `.github/workflows/gpu_tests.yml`
- `.github/workflows/benchmarks.yml`

**完了基準**:
- GPU テストが CI で動作
- パフォーマンス回帰が検出される

---

### Phase 6 の統合テスト

```cpp
TEST(Phase6Integration, RealWorldTransformerTraining) {
    // WMT14 のような実際のデータセットの小規模版で学習
    // ここでは擬似的に
    auto dataset = load_translation_dataset("data/train.txt", src_vocab, tgt_vocab);

    Transformer<float> model(...);
    model.to(DeviceType::CUDA);

    AdamW<float> optimizer(model.parameters(), 0.0001, 0.01);

    for (int epoch = 0; epoch < 10; ++epoch) {
        float total_loss = 0.0;
        int num_batches = 0;

        for (auto& batch : dataset) {
            auto src = batch.src.to(DeviceType::CUDA);
            auto tgt = batch.tgt.to(DeviceType::CUDA);

            auto output = model.forward(src, tgt);
            auto loss = cross_entropy_loss(output, tgt);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_loss += loss.data()[{0}];
            num_batches++;
        }

        float avg_loss = total_loss / num_batches;
        std::cout << "Epoch " << epoch << ", Loss: " << avg_loss << std::endl;

        // 損失が減少していることを確認
        if (epoch > 0) {
            EXPECT_LT(avg_loss, previous_loss);
        }
        previous_loss = avg_loss;
    }

    // 最終的な BLEU スコアを確認
    float bleu = evaluate_bleu(model, test_dataset);
    EXPECT_GT(bleu, 10.0);  // 最低限の BLEU スコア
}
```

---

## Phase 7: 静的計算グラフの実装（オプション）

### 目標

動的グラフに加えて静的グラフを実装し、両方式を比較・学習できるようにする。

### 期間: 2-3 週間

### 前提条件

- Phase 1-6 が完了している
- 動的グラフの実装が十分に理解されている

### 実装項目

#### 7.1 GraphMode の導入 (Week 1)

**実装内容**:
- `GraphMode` enum の追加（`Dynamic` / `Static`）
- `GraphBuilder` クラス: 静的グラフの構築
- 統一インターフェースの設計

**ファイル**:
- `include/gradflow/autograd/graph_mode.hpp`
- `include/gradflow/autograd/graph_builder.hpp`
- `tests/test_graph_mode.cpp`

**テスト項目**:
```cpp
TEST(GraphModeTest, DynamicMode) { ... }
TEST(GraphModeTest, StaticMode) { ... }
TEST(GraphBuilderTest, BuildGraph) { ... }
```

**完了基準**:
- GraphMode の切り替えが動作
- 両モードで基本演算が動作

---

#### 7.2 静的グラフの実装 (Week 1-2)

**実装内容**:
- `StaticGraph` クラス
- グラフの事前構築と最適化
- トポロジカルソート
- メモリ効率の改善

**ファイル**:
- `include/gradflow/autograd/static_graph.hpp`
- `src/autograd/static_graph.cpp`
- `tests/test_static_graph.cpp`

**テスト項目**:
```cpp
TEST(StaticGraphTest, GraphConstruction) { ... }
TEST(StaticGraphTest, ForwardExecution) { ... }
TEST(StaticGraphTest, BackwardExecution) { ... }
```

**完了基準**:
- 静的グラフが構築される
- Forward/Backward が動作
- 動的グラフと同じ結果が得られる

---

#### 7.3 静的グラフの最適化 (Week 2-3)

**実装内容**:
- 定数畳み込み（Constant Folding）
- 共通部分式除去（Common Subexpression Elimination）
- カーネル融合の自動化
- メモリ再利用の最適化

**ファイル**:
- `include/gradflow/autograd/graph_optimizer.hpp`
- `src/autograd/graph_optimizer.cpp`
- `tests/test_graph_optimizer.cpp`

**テスト項目**:
```cpp
TEST(GraphOptimizerTest, ConstantFolding) { ... }
TEST(GraphOptimizerTest, CSE) { ... }
TEST(GraphOptimizerTest, KernelFusion) { ... }
```

**パフォーマンス目標**:
- 静的グラフが動的グラフより 20-30% 高速

**完了基準**:
- 最適化が動作
- パフォーマンスが向上
- 数値的正確性が保たれる

---

### Phase 7 の統合テスト

```cpp
TEST(Phase7Integration, StaticVsDynamicGraph) {
    // 同じモデルを動的・静的両方で実行
    auto X = Tensor<float>::randn({100, 50});
    auto W = Variable<float>::randn({50, 10}, true);

    // 動的グラフ
    set_graph_mode(GraphMode::Dynamic);
    auto y_dynamic = matmul(X, W);
    auto loss_dynamic = y_dynamic.sum();
    loss_dynamic.backward();
    auto grad_dynamic = W.grad().clone();

    // 静的グラフ
    W.zero_grad();
    set_graph_mode(GraphMode::Static);

    // グラフ構築フェーズ
    GraphBuilder builder;
    builder.begin();
    auto y_static = matmul(X, W);
    auto loss_static = y_static.sum();
    auto graph = builder.build();

    // 実行フェーズ
    graph.forward();
    graph.backward();
    auto grad_static = W.grad();

    // 結果が一致することを確認
    EXPECT_TRUE(allclose(y_dynamic, y_static, 1e-5, 1e-7));
    EXPECT_TRUE(allclose(grad_dynamic, grad_static, 1e-5, 1e-7));
}
```

---

## Phase 8: CUDA サポート（オプション）

### 目標

NVIDIA GPU をサポートし、より幅広いハードウェアで実行可能にする。

### 期間: 3-4 週間

### 前提条件

- CUDA Toolkit 12.x がインストールされている
- cuBLAS、cuDNN が利用可能
- Phase 3（Metal）の実装を参考にできる

### 実装項目

#### 8.1 CUDA Device と Allocator (Week 1)

**実装内容**:
- `CUDADevice` 実装
- `CUDAAllocator`: GPU メモリの割り当て
- `CUDAStream` のラッパー
- CPU ↔ GPU のデータ転送

**ファイル**:
- `include/gradflow/autograd/cuda/device.hpp`
- `src/autograd/cuda/device.cu`
- `tests/test_cuda_device.cpp`

**テスト項目**:
```cpp
TEST(CUDADeviceTest, Allocation) { ... }
TEST(CUDADeviceTest, MemoryCopy) { ... }
TEST(CUDADeviceTest, Synchronize) { ... }
```

**完了基準**:
- GPU メモリの割り当てと解放が動作
- CPU ↔ GPU のデータ転送が動作

---

#### 8.2 CUDA カーネルの実装 (Week 1-2)

**実装内容**:
- 要素ごとの演算の CUDA カーネル
- 集約演算の CUDA カーネル
- cuBLAS を使った MatMul
- cuDNN を使った畳み込み

**ファイル**:
- `include/gradflow/autograd/cuda/kernels.hpp`
- `src/autograd/cuda/kernels.cu`
- `tests/test_cuda_kernels.cpp`

**カーネルの例**:
```cuda
template <typename T>
__global__ void add_kernel(const T* a, const T* b, T* c, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

**パフォーマンス目標**:
- MatMul: cuBLAS の 90% 以上の速度
- Tensor Core の活用（Ampere/Hopper）

**完了基準**:
- CPU と GPU の計算結果が一致
- GPU が CPU より高速

---

#### 8.3 CUDA での自動微分 (Week 2-3)

**実装内容**:
- CUDA 上での Operation の実装
- 勾配計算のカーネル

**ファイル**:
- `src/autograd/cuda/ops/*.cu`
- `tests/test_cuda_ops_grad.cpp`

**テスト項目**:
```cpp
TEST(CUDAOpsGradTest, MatMulGradient) { ... }
TEST(CUDAOpsGradTest, ReLUGradient) { ... }
```

**完了基準**:
- CUDA GPU での勾配計算が CPU と一致
- 数値勾配チェックがすべてパス

---

#### 8.4 Metal と CUDA の統一 (Week 3-4)

**実装内容**:
- Device 抽象化の完成
- Metal と CUDA の切り替えが透過的に
- パフォーマンス比較

**完了基準**:
- 同じコードが Metal と CUDA の両方で動作
- デバイス固有の最適化が適用される

---

### Phase 8 の統合テスト

```cpp
TEST(Phase8Integration, MultiGPUSupport) {
    // Metal と CUDA の両方で同じ計算を実行
    std::vector<DeviceType> devices = {DeviceType::Metal, DeviceType::CUDA};

    for (auto device : devices) {
        if (!is_available(device)) continue;

        auto X = Variable<float>::randn({1000, 512}, true, device);
        auto W = Variable<float>::randn({512, 256}, true, device);

        auto y = matmul(X, W);
        auto loss = y.sum();
        loss.backward();

        EXPECT_GT(W.grad().abs().sum().data()[{0}], 0.0);
        std::cout << device_name(device) << " test passed" << std::endl;
    }
}
```

---

## 継続的な改善

Phase 6 以降も、以下の改善を継続的に実施します。

### 追加実装項目（Phase 7-8 完了後）

### 1. パフォーマンス最適化

- **FlashAttention の実装**: メモリアクセスを最適化した Attention
- **Kernel 融合の拡大**: より多くの演算を融合
- **混合精度学習**: FP16/BF16 のサポート

### 2. 新機能の追加

- **他のアーキテクチャ**: GPT, BERT, Vision Transformer
- **分散学習**: Data Parallel, Model Parallel
- **量子化**: INT8 推論

### 3. ユーザビリティの向上

- **より良いエラーメッセージ**: デバッグを容易にする
- **計算グラフの可視化**: Graphviz などで可視化
- **プロファイリングツール**: ボトルネックの特定

### 4. エコシステムの拡大

- **モデルズー**: 事前学習済みモデルの提供
- **データローダー**: 効率的なデータロード
- **コミュニティの構築**: Issue, PR の活性化

---

## マイルストーン

| フェーズ | 完了時期 | 主要な成果物 |
|---------|---------|------------|
| Phase 1 | Week 6 | Tensor, CPU 演算（動的グラフ） |
| Phase 2 | Week 12 | 自動微分, Optimizer（動的グラフ） |
| Phase 3 | Week 16 | Metal GPU サポート |
| Phase 4 | Week 20 | 最適化, LayerNorm, Embedding |
| Phase 5 | Week 25 | Transformer 完全実装 |
| Phase 6 | Week 28 | 統合, ベンチマーク, ドキュメント |
| Phase 7 | Week 31 | 静的グラフ実装（オプション） |
| Phase 8 | Week 35 | CUDA サポート（オプション） |

---

## リスクと緩和策

### リスク 1: GPU バックエンドの複雑性

**リスク**: Metal/CUDA の実装が予想以上に複雑で時間がかかる

**緩和策**:
- Phase 3 で Metal Performance Shaders (MPS) などの既存ライブラリを活用
- Phase 8 で cuBLAS/cuDNN などの成熟したライブラリを活用
- 単純なカーネルから始めて段階的に最適化
- GPU なしでも動作するように CPU フォールバックを実装

### リスク 2: パフォーマンス目標の未達

**リスク**: 目標とするパフォーマンスが達成できない

**緩和策**:
- 早期にベンチマークを実施してボトルネックを特定
- プロファイリングツール（Xcode Instruments, nvprof, perf）を活用
- Metal では Unified Memory の利点を最大限活用
- CUDA では Tensor Core の活用を検討
- コミュニティからフィードバックを得る

### リスク 3: メモリリーク

**リスク**: 複雑な計算グラフでメモリリークが発生

**緩和策**:
- RAII を徹底して使用
- Valgrind, AddressSanitizer による定期的なチェック
- 参照カウントやスマートポインタの適切な使用

### リスク 4: 数値不安定性

**リスク**: 勾配爆発/消失、Softmax のオーバーフローなど

**緩和策**:
- 数値安定性のテストを充実させる
- Gradient clipping の実装
- PyTorch の実装を参考にする

---

## まとめ

本ロードマップに従って開発を進めることで、教育的でありながら実用的な自動微分ライブラリと Transformer 実装を構築できます。各フェーズは独立してテスト可能であり、段階的に機能を追加していくことで、リスクを最小化しながら目標を達成できます。

次のステップは、[技術選定の理由書](./TECHNICAL_DECISIONS.md) を参照してください。
