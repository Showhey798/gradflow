# Autodiff Library & Transformer Architecture Design

## 目次
- [概要](#概要)
- [設計哲学](#設計哲学)
- [システムアーキテクチャ](#システムアーキテクチャ)
- [コアコンポーネント](#コアコンポーネント)
- [計算グラフと自動微分](#計算グラフと自動微分)
- [デバイス抽象化](#デバイス抽象化)
- [メモリ管理](#メモリ管理)
- [演算レイヤー](#演算レイヤー)
- [Transformer コンポーネント](#transformer-コンポーネント)
- [最適化とパフォーマンス](#最適化とパフォーマンス)

---

## 概要

本ドキュメントは、**GradFlow** における自動微分ライブラリと Transformer 実装のアーキテクチャを定義します。このライブラリは PyTorch の動的計算グラフの柔軟性と、CUTLASS/CuTe のような低レイヤーの効率性を兼ね備えた設計を目指します。

### 設計目標

1. **教育的明瞭性**: アルゴリズムの各ステップが追跡可能で理解しやすい
2. **パフォーマンス**: CPU/Metal GPU/CUDA GPU での高効率な計算
3. **拡張性**: 新しい演算・レイヤーの追加が容易
4. **型安全性**: C++17 テンプレートによるコンパイル時の保証
5. **Python 統合**: nanobind 経由でシームレスな Python インターフェース
6. **両方式サポート**: 動的・静的計算グラフの両方を実装し学習

---

## 設計哲学

### SOLID 原則の適用

#### 単一責任の原則 (Single Responsibility)
- `Tensor` クラス: データの保持とアクセスのみ
- `Operation` クラス: 特定の演算の forward/backward のみ
- `Device` クラス: デバイス固有の計算リソース管理のみ

#### 開放閉鎖の原則 (Open/Closed)
- 基底クラス `Operation` を継承することで、新しい演算を追加可能
- `Device` インターフェースにより、新しいハードウェアバックエンドを追加可能

#### リスコフ置換の原則 (Liskov Substitution)
- すべての `Operation` サブクラスは基底クラスと互換性を持つ
- すべての `Device` 実装は同じインターフェースを提供

#### インターフェース分離の原則 (Interface Segregation)
- `IForwardOperation`: forward のみが必要な演算
- `IBackwardOperation`: backward も必要な演算
- `IInplaceOperation`: インプレース演算が可能な演算

#### 依存性逆転の原則 (Dependency Inversion)
- 高レベルモジュール（Transformer）は抽象インターフェース（Operation）に依存
- 低レベルモジュール（MatMul, Softmax）は抽象インターフェースを実装

### 動的計算グラフ vs 静的計算グラフ

本ライブラリは **両方の計算グラフ方式を実装**します。

#### Phase 1-6: 動的計算グラフ（Define-by-Run）

**採用理由**:
1. **柔軟性**: Python の制御フロー（if/for/while）をそのまま使用可能
2. **デバッグ性**: 計算の各ステップをリアルタイムで検証可能
3. **実装の簡潔さ**: グラフの事前定義が不要
4. **教育的価値**: 処理フローが直感的で理解しやすい

**トレードオフ**:
- **実行時オーバーヘッド**: グラフ構築のコストが毎回発生
- **最適化の制限**: 静的グラフのようなグローバル最適化が困難

**緩和策**:
- **JIT コンパイル**: 頻繁に実行されるサブグラフをキャッシュ
- **演算融合**: 連続する単純な演算を一つのカーネルに統合
- **メモリプール**: テンソルの再割り当てコストを削減

#### Phase 7: 静的計算グラフ（Define-and-Run）

**追加理由**:
1. **学習目的**: 両方式の設計を深く理解
2. **パフォーマンス**: グローバル最適化により 20-30% 高速化
3. **最適化**: 定数畳み込み、共通部分式除去、カーネル融合の自動化
4. **メモリ効率**: 事前にメモリ配置を最適化

**実装方針**:
- `GraphMode` enum により動的・静的を切り替え可能
- 両モードで数値的に同一の結果を保証
- 統一インターフェースで既存コードの変更を最小限に

参考: [PyTorch の動的計算グラフ](https://pytorch.org/blog/computational-graphs-constructed-in-pytorch/)、[TensorFlow の静的グラフ最適化](https://www.tensorflow.org/guide/graph_optimization)

---

## システムアーキテクチャ

### レイヤー構造

```
┌─────────────────────────────────────────────────────────┐
│          High-Level API (Python/C++)                     │
│  (Transformer, Encoder, Decoder, MultiHeadAttention)     │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│         Neural Network Components                        │
│  (Layer Normalization, Dropout, Embedding)               │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│         Autograd Engine (Computational Graph)            │
│  (Variable, Operation, GradientTape)                     │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│         Tensor Operations Layer                          │
│  (MatMul, Add, ReLU, Softmax, etc.)                      │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│         Tensor Abstraction                               │
│  (Tensor<T>, Shape, Stride, Storage)                     │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│         Device Abstraction Layer                         │
│  (CPU, CUDA, Metal, OpenCL)                              │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│         Memory Management                                │
│  (MemoryPool, Allocator, DeviceBuffer)                   │
└─────────────────────────────────────────────────────────┘
```

### モジュール間の依存関係

```
Transformer
    ↓
MultiHeadAttention → LayerNorm → PositionwiseFeedForward
    ↓                   ↓                ↓
  Softmax             Add/Mul          GELU
    ↓                   ↓                ↓
  MatMul  ←──────── Variable ──────→  Add
    ↓                   ↓
  Tensor ←────────── Operation
    ↓                   ↓
  Device ←────────── MemoryPool
```

---

## コアコンポーネント

### 1. Tensor クラス

Tensor は多次元配列の抽象化であり、データ、形状、ストライド、デバイス情報を保持します。

#### 設計仕様

```cpp
namespace fullscratch {
namespace autograd {

template <typename T>
class Tensor {
public:
    // コンストラクタ
    Tensor(const Shape& shape, DeviceType device = DeviceType::CPU);
    Tensor(const Shape& shape, const std::vector<T>& data, DeviceType device = DeviceType::CPU);

    // データアクセス
    T& operator[](const std::vector<size_t>& indices);
    const T& operator[](const std::vector<size_t>& indices) const;

    // プロパティ
    const Shape& shape() const;
    const Stride& stride() const;
    size_t ndim() const;
    size_t size() const;
    DeviceType device() const;

    // デバイス間転送
    Tensor<T> to(DeviceType device) const;

    // ビュー操作（メモリコピーなし）
    Tensor<T> view(const Shape& new_shape) const;
    Tensor<T> reshape(const Shape& new_shape) const;
    Tensor<T> transpose(size_t dim0, size_t dim1) const;

    // メモリ管理
    T* data();
    const T* data() const;
    bool is_contiguous() const;
    Tensor<T> contiguous() const;

private:
    std::shared_ptr<Storage<T>> storage_;
    Shape shape_;
    Stride stride_;
    size_t offset_;
    DeviceType device_;
};

} // namespace autograd
} // namespace gradflow
```

#### 重要な設計判断

##### メモリレイアウト: Row-Major

理由:
- C++ の標準配列と互換性がある
- NumPy と同じレイアウト（Python 統合が容易）
- CPU キャッシュ効率が高い（行方向のアクセスパターン）

##### Stride（ストライド）の導入

各次元での要素間の間隔を保持することで、以下が可能になります:
- **ゼロコピーのビュー操作**: `transpose()`, `view()` がメモリコピー不要
- **Broadcasting**: 形状が異なるテンソル間の演算
- **スライシング**: 部分テンソルの効率的な抽出

例:
```cpp
// 元のテンソル: shape=[2, 3], stride=[3, 1]
// [1, 2, 3]
// [4, 5, 6]

// 転置後: shape=[3, 2], stride=[1, 3] (データは同じ)
// [1, 4]
// [2, 5]
// [3, 6]
```

参考: [PyTorch Tensor Internals](https://docs.pytorch.org/docs/stable/notes/cuda.html)

---

### 2. Shape と Stride

```cpp
class Shape {
public:
    Shape(std::initializer_list<size_t> dims);
    Shape(const std::vector<size_t>& dims);

    size_t ndim() const;
    size_t size() const;  // 全要素数
    const std::vector<size_t>& dims() const;
    size_t operator[](size_t idx) const;

    bool operator==(const Shape& other) const;
    bool is_broadcastable_to(const Shape& other) const;

private:
    std::vector<size_t> dims_;
};

class Stride {
public:
    Stride(const Shape& shape);  // Row-major stride を計算
    Stride(const std::vector<size_t>& strides);

    size_t offset(const std::vector<size_t>& indices) const;
    const std::vector<size_t>& strides() const;

private:
    std::vector<size_t> strides_;
};
```

---

### 3. Storage クラス

実際のメモリバッファを管理します。

```cpp
template <typename T>
class Storage {
public:
    Storage(size_t size, DeviceType device);
    ~Storage();

    T* data();
    const T* data() const;
    size_t size() const;
    DeviceType device() const;

    // デバイス間のコピー
    std::shared_ptr<Storage<T>> to(DeviceType device) const;

private:
    T* data_;
    size_t size_;
    DeviceType device_;
    std::shared_ptr<DeviceAllocator> allocator_;
};
```

---

## 計算グラフと自動微分

### Variable クラス

`Variable` は `Tensor` をラップし、勾配計算のための情報を保持します。

```cpp
template <typename T>
class Variable {
public:
    Variable(const Tensor<T>& data, bool requires_grad = false);

    // データアクセス
    Tensor<T>& data();
    const Tensor<T>& data() const;

    // 勾配
    Tensor<T>& grad();
    const Tensor<T>& grad() const;
    bool requires_grad() const;

    // 計算グラフ
    std::shared_ptr<Operation<T>> grad_fn() const;
    void set_grad_fn(std::shared_ptr<Operation<T>> fn);

    // 逆伝播
    void backward(const Tensor<T>& grad = Tensor<T>::ones_like(data_));

    // 勾配のゼロクリア
    void zero_grad();

private:
    Tensor<T> data_;
    Tensor<T> grad_;
    bool requires_grad_;
    std::shared_ptr<Operation<T>> grad_fn_;
};
```

### Operation 基底クラス

すべての演算は `Operation` を継承します。

```cpp
template <typename T>
class Operation : public std::enable_shared_from_this<Operation<T>> {
public:
    virtual ~Operation() = default;

    // Forward pass（サブクラスで実装）
    virtual Tensor<T> forward(const std::vector<Tensor<T>>& inputs) = 0;

    // Backward pass（サブクラスで実装）
    // 出力の勾配を入力として受け取り、入力の勾配を返す
    virtual std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) = 0;

    // 入力変数（勾配計算のために保存）
    std::vector<Variable<T>> inputs() const { return inputs_; }

protected:
    std::vector<Variable<T>> inputs_;

    // Forward pass で保存が必要な中間変数
    std::unordered_map<std::string, Tensor<T>> saved_tensors_;

    void saveForBackward(const std::string& name, const Tensor<T>& tensor) {
        saved_tensors_[name] = tensor;
    }

    Tensor<T> getSavedTensor(const std::string& name) const {
        return saved_tensors_.at(name);
    }
};
```

### 計算グラフの構築と実行

#### Forward Pass

```cpp
// 例: z = (x * y) + b
auto x = Variable<float>(Tensor<float>({2, 3}), true);
auto y = Variable<float>(Tensor<float>({2, 3}), true);
auto b = Variable<float>(Tensor<float>({2, 3}), true);

// Forward pass（自動的に計算グラフが構築される）
auto mul_result = x * y;  // MulOperation を生成
auto z = mul_result + b;  // AddOperation を生成

// 計算グラフ構造:
// x ───┐
//      ├─→ [Mul] ─→ mul_result ─┐
// y ───┘                         ├─→ [Add] ─→ z
// b ─────────────────────────────┘
```

#### Backward Pass

```cpp
// Backward pass（自動微分）
z.backward();

// 勾配が各変数に蓄積される
auto dx = x.grad();  // ∂z/∂x
auto dy = y.grad();  // ∂z/∂y
auto db = b.grad();  // ∂z/∂b
```

#### 実装の詳細

```cpp
template <typename T>
void Variable<T>::backward(const Tensor<T>& grad) {
    // 勾配の初期化
    if (grad_.size() == 0) {
        grad_ = Tensor<T>::zeros_like(data_);
    }

    // 勾配の蓄積
    grad_ = grad_ + grad;

    // この変数が演算の出力である場合、逆伝播を継続
    if (grad_fn_) {
        auto input_grads = grad_fn_->backward(grad);
        auto inputs = grad_fn_->inputs();

        for (size_t i = 0; i < inputs.size(); ++i) {
            if (inputs[i].requires_grad()) {
                inputs[i].backward(input_grads[i]);
            }
        }
    }
}
```

参考:
- [PyTorch Autograd in C++](https://docs.pytorch.org/tutorials/advanced/cpp_autograd.html)
- [Dynamic vs Static Computational Graphs](https://towardsdatascience.com/computational-graphs-in-pytorch-and-tensorflow-c25cc40bdcd1/)

---

## デバイス抽象化

### DeviceType 列挙型

```cpp
enum class DeviceType {
    CPU,
    CUDA,
    Metal,
    OpenCL
};
```

### Device インターフェース

```cpp
class Device {
public:
    virtual ~Device() = default;

    virtual DeviceType type() const = 0;
    virtual int id() const = 0;

    // メモリ操作
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual void memcpy(void* dst, const void* src, size_t bytes) = 0;
    virtual void memset(void* ptr, int value, size_t bytes) = 0;

    // 同期
    virtual void synchronize() = 0;

    // プロパティ
    virtual size_t total_memory() const = 0;
    virtual size_t available_memory() const = 0;
};
```

### CPU Device 実装

```cpp
class CPUDevice : public Device {
public:
    DeviceType type() const override { return DeviceType::CPU; }
    int id() const override { return 0; }

    void* allocate(size_t bytes) override {
        return std::aligned_alloc(64, bytes);  // 64-byte alignment for SIMD
    }

    void deallocate(void* ptr) override {
        std::free(ptr);
    }

    void memcpy(void* dst, const void* src, size_t bytes) override {
        std::memcpy(dst, src, bytes);
    }

    void memset(void* ptr, int value, size_t bytes) override {
        std::memset(ptr, value, bytes);
    }

    void synchronize() override {
        // CPU は同期不要
    }

    size_t total_memory() const override;
    size_t available_memory() const override;
};
```

### CUDA Device 実装（概要）

```cpp
class CUDADevice : public Device {
public:
    CUDADevice(int device_id);

    DeviceType type() const override { return DeviceType::CUDA; }
    int id() const override { return device_id_; }

    void* allocate(size_t bytes) override {
        void* ptr;
        cudaMalloc(&ptr, bytes);
        return ptr;
    }

    void deallocate(void* ptr) override {
        cudaFree(ptr);
    }

    void memcpy(void* dst, const void* src, size_t bytes) override {
        cudaMemcpy(dst, src, bytes, cudaMemcpyDefault);
    }

    void synchronize() override {
        cudaDeviceSynchronize();
    }

    // CUDA 固有の機能
    cudaStream_t stream() const { return stream_; }

private:
    int device_id_;
    cudaStream_t stream_;
};
```

### デバイス間のデータ転送

```cpp
template <typename T>
Tensor<T> Tensor<T>::to(DeviceType target_device) const {
    if (device_ == target_device) {
        return *this;  // 既に目的のデバイス上にある
    }

    // 新しいストレージを作成
    auto new_storage = std::make_shared<Storage<T>>(size(), target_device);

    // デバイス間でデータをコピー
    auto src_device = DeviceManager::get(device_);
    auto dst_device = DeviceManager::get(target_device);

    // CPU ↔ GPU のような異種デバイス間のコピーは
    // 一時的に CPU バッファを経由する場合がある
    if (device_ == DeviceType::CPU || target_device == DeviceType::CPU) {
        dst_device->memcpy(new_storage->data(), storage_->data(), size() * sizeof(T));
    } else {
        // GPU → GPU の場合は peer-to-peer コピーを試みる
        // 実装は省略
    }

    return Tensor<T>(new_storage, shape_, stride_, 0, target_device);
}
```

参考: [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

---

## メモリ管理

### MemoryPool 設計

GPU メモリのアロケーション/デアロケーションは非常にコストが高いため、メモリプールを使用します。

```cpp
class MemoryPool {
public:
    MemoryPool(DeviceType device, size_t initial_size = 1024 * 1024 * 1024);  // 1GB

    void* allocate(size_t bytes);
    void deallocate(void* ptr);

    // メモリプールの統計情報
    size_t total_allocated() const;
    size_t total_reserved() const;
    size_t available() const;

    // メモリの断片化を解消
    void defragment();

private:
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
    };

    DeviceType device_;
    std::shared_ptr<Device> device_impl_;
    std::vector<Block> blocks_;

    // Best-fit アロケーション戦略
    Block* find_best_fit(size_t bytes);
    void split_block(Block* block, size_t bytes);
    void merge_free_blocks();
};
```

### アロケーション戦略

#### Best-Fit アルゴリズム

要求されたサイズに最も近い空きブロックを選択します。

```cpp
MemoryPool::Block* MemoryPool::find_best_fit(size_t bytes) {
    Block* best = nullptr;
    size_t min_waste = std::numeric_limits<size_t>::max();

    for (auto& block : blocks_) {
        if (!block.in_use && block.size >= bytes) {
            size_t waste = block.size - bytes;
            if (waste < min_waste) {
                min_waste = waste;
                best = &block;
            }
        }
    }

    return best;
}
```

#### 断片化の緩和

定期的に隣接する空きブロックをマージします。

```cpp
void MemoryPool::merge_free_blocks() {
    std::sort(blocks_.begin(), blocks_.end(),
        [](const Block& a, const Block& b) { return a.ptr < b.ptr; });

    for (size_t i = 0; i < blocks_.size() - 1; ++i) {
        if (!blocks_[i].in_use && !blocks_[i + 1].in_use) {
            blocks_[i].size += blocks_[i + 1].size;
            blocks_.erase(blocks_.begin() + i + 1);
            --i;
        }
    }
}
```

参考:
- [Capuchin: Tensor-based GPU Memory Management](https://dl.acm.org/doi/10.1145/3373376.3378505)
- [PyTorch CUDA Semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)

---

## 演算レイヤー

### 基本演算

#### MatMul（行列積）

```cpp
template <typename T>
class MatMulOperation : public Operation<T> {
public:
    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        const auto& a = inputs[0];
        const auto& b = inputs[1];

        // a: [M, K], b: [K, N] → result: [M, N]
        size_t M = a.shape()[0];
        size_t K = a.shape()[1];
        size_t N = b.shape()[1];

        Tensor<T> result({M, N}, a.device());

        if (a.device() == DeviceType::CPU) {
            // CPU 実装: BLAS ライブラリを使用
            cpu_matmul(a.data(), b.data(), result.data(), M, K, N);
        } else if (a.device() == DeviceType::CUDA) {
            // GPU 実装: cuBLAS を使用
            cuda_matmul(a.data(), b.data(), result.data(), M, K, N);
        }

        // Backward pass のために入力を保存
        saveForBackward("a", a);
        saveForBackward("b", b);

        return result;
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto a = getSavedTensor("a");
        auto b = getSavedTensor("b");

        // ∂L/∂A = ∂L/∂C @ B^T
        Tensor<T> grad_a = matmul(grad_output, b.transpose(-1, -2));

        // ∂L/∂B = A^T @ ∂L/∂C
        Tensor<T> grad_b = matmul(a.transpose(-1, -2), grad_output);

        return {grad_a, grad_b};
    }
};
```

#### Softmax

```cpp
template <typename T>
class SoftmaxOperation : public Operation<T> {
public:
    SoftmaxOperation(int dim) : dim_(dim) {}

    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        const auto& x = inputs[0];

        // Numerical stability: x - max(x)
        auto x_max = x.max(dim_, /*keepdim=*/true);
        auto x_shifted = x - x_max;

        // exp(x)
        auto exp_x = x_shifted.exp();

        // sum(exp(x))
        auto sum_exp = exp_x.sum(dim_, /*keepdim=*/true);

        // softmax = exp(x) / sum(exp(x))
        auto result = exp_x / sum_exp;

        saveForBackward("softmax", result);

        return result;
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto softmax = getSavedTensor("softmax");

        // ∂softmax/∂x = softmax * (grad_output - (grad_output * softmax).sum(dim))
        auto sum_term = (grad_output * softmax).sum(dim_, /*keepdim=*/true);
        auto grad_input = softmax * (grad_output - sum_term);

        return {grad_input};
    }

private:
    int dim_;
};
```

#### LayerNorm

```cpp
template <typename T>
class LayerNormOperation : public Operation<T> {
public:
    LayerNormOperation(const std::vector<size_t>& normalized_shape, T eps = 1e-5)
        : normalized_shape_(normalized_shape), eps_(eps) {}

    Tensor<T> forward(const std::vector<Tensor<T>>& inputs) override {
        const auto& x = inputs[0];
        const auto& gamma = inputs[1];  // scale
        const auto& beta = inputs[2];   // shift

        // 正規化する次元を決定
        int dims_to_normalize = normalized_shape_.size();
        std::vector<int> axes;
        for (int i = 0; i < dims_to_normalize; ++i) {
            axes.push_back(x.ndim() - dims_to_normalize + i);
        }

        // Mean and variance
        auto mean = x.mean(axes, /*keepdim=*/true);
        auto variance = x.var(axes, /*keepdim=*/true, /*unbiased=*/false);

        // Normalize
        auto x_normalized = (x - mean) / (variance + eps_).sqrt();

        // Scale and shift
        auto result = gamma * x_normalized + beta;

        // Save for backward
        saveForBackward("x_normalized", x_normalized);
        saveForBackward("gamma", gamma);
        saveForBackward("std", (variance + eps_).sqrt());

        return result;
    }

    std::vector<Tensor<T>> backward(const Tensor<T>& grad_output) override {
        auto x_normalized = getSavedTensor("x_normalized");
        auto gamma = getSavedTensor("gamma");
        auto std = getSavedTensor("std");

        // ∂L/∂gamma
        auto grad_gamma = (grad_output * x_normalized).sum(/*keep dims except normalized*/);

        // ∂L/∂beta
        auto grad_beta = grad_output.sum(/*keep dims except normalized*/);

        // ∂L/∂x (複雑な導出は省略)
        auto grad_x_normalized = grad_output * gamma;
        auto grad_x = grad_x_normalized / std;  // 簡略版

        return {grad_x, grad_gamma, grad_beta};
    }

private:
    std::vector<size_t> normalized_shape_;
    T eps_;
};
```

---

## Transformer コンポーネント

### Scaled Dot-Product Attention

```cpp
template <typename T>
class ScaledDotProductAttention {
public:
    ScaledDotProductAttention(T dropout = 0.0) : dropout_(dropout) {}

    Variable<T> forward(const Variable<T>& query,
                        const Variable<T>& key,
                        const Variable<T>& value,
                        const Variable<T>* mask = nullptr) {
        // query: [batch, num_heads, seq_len, d_k]
        // key:   [batch, num_heads, seq_len, d_k]
        // value: [batch, num_heads, seq_len, d_v]

        T d_k = static_cast<T>(query.data().shape()[-1]);
        T scale = 1.0 / std::sqrt(d_k);

        // Attention scores: Q @ K^T / sqrt(d_k)
        auto key_transposed = key.transpose(-2, -1);
        auto scores = matmul(query, key_transposed) * scale;

        // Apply mask (if provided)
        if (mask != nullptr) {
            scores = scores + (*mask) * static_cast<T>(-1e9);
        }

        // Softmax
        auto attn_weights = softmax(scores, /*dim=*/-1);

        // Apply dropout
        if (dropout_ > 0.0 && is_training_) {
            attn_weights = dropout(attn_weights, dropout_);
        }

        // Weighted sum: Attention @ V
        auto output = matmul(attn_weights, value);

        return output;
    }

private:
    T dropout_;
    bool is_training_ = true;
};
```

### Multi-Head Attention

```cpp
template <typename T>
class MultiHeadAttention {
public:
    MultiHeadAttention(size_t d_model, size_t num_heads, T dropout = 0.0)
        : d_model_(d_model), num_heads_(num_heads), dropout_(dropout) {

        assert(d_model % num_heads == 0);
        d_k_ = d_model / num_heads;

        // Linear projections
        W_q_ = Variable<T>(Tensor<T>::randn({d_model, d_model}), true);
        W_k_ = Variable<T>(Tensor<T>::randn({d_model, d_model}), true);
        W_v_ = Variable<T>(Tensor<T>::randn({d_model, d_model}), true);
        W_o_ = Variable<T>(Tensor<T>::randn({d_model, d_model}), true);

        attention_ = std::make_unique<ScaledDotProductAttention<T>>(dropout);
    }

    Variable<T> forward(const Variable<T>& query,
                        const Variable<T>& key,
                        const Variable<T>& value,
                        const Variable<T>* mask = nullptr) {
        size_t batch_size = query.data().shape()[0];
        size_t seq_len = query.data().shape()[1];

        // Linear projections: [batch, seq_len, d_model] -> [batch, seq_len, d_model]
        auto Q = linear(query, W_q_);
        auto K = linear(key, W_k_);
        auto V = linear(value, W_v_);

        // Reshape for multi-head: [batch, seq_len, d_model]
        //                      -> [batch, num_heads, seq_len, d_k]
        Q = Q.view({batch_size, seq_len, num_heads_, d_k_}).transpose(1, 2);
        K = K.view({batch_size, seq_len, num_heads_, d_k_}).transpose(1, 2);
        V = V.view({batch_size, seq_len, num_heads_, d_k_}).transpose(1, 2);

        // Scaled dot-product attention
        auto attn_output = attention_->forward(Q, K, V, mask);

        // Concatenate heads: [batch, num_heads, seq_len, d_k]
        //                 -> [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous();
        attn_output = attn_output.view({batch_size, seq_len, d_model_});

        // Final linear projection
        auto output = linear(attn_output, W_o_);

        return output;
    }

private:
    size_t d_model_;
    size_t num_heads_;
    size_t d_k_;
    T dropout_;

    Variable<T> W_q_, W_k_, W_v_, W_o_;
    std::unique_ptr<ScaledDotProductAttention<T>> attention_;
};
```

### Position-wise Feed-Forward Network

```cpp
template <typename T>
class PositionwiseFeedForward {
public:
    PositionwiseFeedForward(size_t d_model, size_t d_ff, T dropout = 0.0)
        : d_model_(d_model), d_ff_(d_ff), dropout_(dropout) {

        W1_ = Variable<T>(Tensor<T>::randn({d_model, d_ff}), true);
        b1_ = Variable<T>(Tensor<T>::zeros({d_ff}), true);
        W2_ = Variable<T>(Tensor<T>::randn({d_ff, d_model}), true);
        b2_ = Variable<T>(Tensor<T>::zeros({d_model}), true);
    }

    Variable<T> forward(const Variable<T>& x) {
        // FFN(x) = max(0, xW1 + b1)W2 + b2
        auto hidden = gelu(linear(x, W1_, b1_));

        if (dropout_ > 0.0 && is_training_) {
            hidden = dropout(hidden, dropout_);
        }

        auto output = linear(hidden, W2_, b2_);

        return output;
    }

private:
    size_t d_model_;
    size_t d_ff_;
    T dropout_;
    bool is_training_ = true;

    Variable<T> W1_, b1_, W2_, b2_;
};
```

### Encoder Layer

```cpp
template <typename T>
class TransformerEncoderLayer {
public:
    TransformerEncoderLayer(size_t d_model, size_t num_heads, size_t d_ff, T dropout = 0.1)
        : d_model_(d_model) {

        self_attn_ = std::make_unique<MultiHeadAttention<T>>(d_model, num_heads, dropout);
        feed_forward_ = std::make_unique<PositionwiseFeedForward<T>>(d_model, d_ff, dropout);

        norm1_ = std::make_unique<LayerNorm<T>>(std::vector<size_t>{d_model});
        norm2_ = std::make_unique<LayerNorm<T>>(std::vector<size_t>{d_model});

        dropout_ = dropout;
    }

    Variable<T> forward(const Variable<T>& x, const Variable<T>* mask = nullptr) {
        // Self-attention with residual connection and layer norm
        auto attn_output = self_attn_->forward(x, x, x, mask);

        if (dropout_ > 0.0 && is_training_) {
            attn_output = dropout(attn_output, dropout_);
        }

        auto x1 = norm1_->forward(x + attn_output);

        // Feed-forward with residual connection and layer norm
        auto ff_output = feed_forward_->forward(x1);

        if (dropout_ > 0.0 && is_training_) {
            ff_output = dropout(ff_output, dropout_);
        }

        auto x2 = norm2_->forward(x1 + ff_output);

        return x2;
    }

private:
    size_t d_model_;
    T dropout_;
    bool is_training_ = true;

    std::unique_ptr<MultiHeadAttention<T>> self_attn_;
    std::unique_ptr<PositionwiseFeedForward<T>> feed_forward_;
    std::unique_ptr<LayerNorm<T>> norm1_;
    std::unique_ptr<LayerNorm<T>> norm2_;
};
```

### Positional Encoding

```cpp
template <typename T>
class PositionalEncoding {
public:
    PositionalEncoding(size_t d_model, size_t max_len = 5000) {
        // PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        // PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        Tensor<T> pe({max_len, d_model});

        for (size_t pos = 0; pos < max_len; ++pos) {
            for (size_t i = 0; i < d_model; i += 2) {
                T angle = static_cast<T>(pos) / std::pow(10000.0, static_cast<T>(i) / d_model);
                pe[{pos, i}] = std::sin(angle);
                if (i + 1 < d_model) {
                    pe[{pos, i + 1}] = std::cos(angle);
                }
            }
        }

        pe_ = Variable<T>(pe, false);  // requires_grad=false
    }

    Variable<T> forward(const Variable<T>& x) {
        // x: [batch, seq_len, d_model]
        size_t seq_len = x.data().shape()[1];

        // Add positional encoding
        auto pe_slice = pe_.data().slice(0, 0, seq_len);
        return x + Variable<T>(pe_slice, false);
    }

private:
    Variable<T> pe_;
};
```

参考:
- [Attention Is All You Need (Original Paper)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

---

## 最適化とパフォーマンス

### CPU 最適化

#### SIMD ベクトル化

```cpp
// Example: Vectorized addition with AVX2
void vector_add_avx2(const float* a, const float* b, float* result, size_t n) {
    size_t i = 0;

    // Process 8 floats at a time with AVX2
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(result + i, vr);
    }

    // Handle remaining elements
    for (; i < n; ++i) {
        result[i] = a[i] + b[i];
    }
}
```

#### OpenMP による並列化

```cpp
void matmul_parallel(const float* A, const float* B, float* C,
                     size_t M, size_t K, size_t N) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
```

### GPU 最適化

#### Kernel Fusion（演算融合）

複数の演算を一つの CUDA カーネルに統合して、メモリアクセスを削減します。

```cpp
// Before fusion:
auto x1 = x + y;      // Kernel 1
auto x2 = x1 * z;     // Kernel 2
auto x3 = relu(x2);   // Kernel 3

// After fusion:
auto x3 = fused_add_mul_relu(x, y, z);  // Single kernel
```

#### FlashAttention

メモリアクセスを最適化した Attention の実装。

参考: [FlashAttention Paper](https://arxiv.org/abs/2205.14135)

```cpp
// Tiling を使用してメモリアクセスを最適化
// 詳細な実装は複雑なため、概念のみを示す
template <typename T>
Tensor<T> flash_attention(const Tensor<T>& Q, const Tensor<T>& K, const Tensor<T>& V) {
    // SRAM（共有メモリ）に収まるサイズのタイルに分割
    constexpr size_t TILE_SIZE = 64;

    // Outer loop: Q をタイル単位で処理
    // Inner loop: K, V をタイル単位でストリーミング

    // 各タイルで Attention を計算し、オンラインで Softmax を更新
    // これにより、巨大な Attention 行列をメモリに保持せずに済む
}
```

### メモリ最適化

#### Gradient Checkpointing

メモリ使用量を削減するために、一部の中間層の出力を保存せず、逆伝播時に再計算します。

```cpp
template <typename T>
class CheckpointedLayer {
public:
    Variable<T> forward(const Variable<T>& x) {
        // Forward pass の中間結果を保存しない
        auto y = layer_->forward(x);

        // 入力のみを保存
        saved_input_ = x;

        return y;
    }

    Variable<T> backward(const Variable<T>& grad_output) {
        // Forward pass を再実行
        auto y = layer_->forward(saved_input_);

        // Backward pass
        return layer_->backward(grad_output);
    }

private:
    std::unique_ptr<Layer<T>> layer_;
    Variable<T> saved_input_;
};
```

#### Inplace Operations

可能な場合は、新しいテンソルを割り当てずに既存のテンソルを更新します。

```cpp
// Not inplace
auto y = x + 1;  // 新しいテンソルを割り当て

// Inplace
x.add_(1);  // x を直接更新
```

---

## まとめ

本アーキテクチャは、以下の原則に基づいて設計されています:

1. **モジュール性**: 各コンポーネントは独立してテスト・拡張可能
2. **抽象化**: デバイス、演算、メモリ管理が適切に抽象化されている
3. **パフォーマンス**: CPU/GPU での効率的な計算を実現
4. **拡張性**: 新しい演算やデバイスの追加が容易
5. **教育的価値**: 各アルゴリズムの動作が明確に追跡可能

次のステップは、[API 設計書](./API_DESIGN.md) と [実装ロードマップ](./ROADMAP.md) を参照してください。
