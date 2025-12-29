# API Design Document

## 目次
- [概要](#概要)
- [C++ API](#c-api)
- [Python API](#python-api)
- [使用例](#使用例)
- [API リファレンス](#api-リファレンス)

---

## 概要

fullScratchLibs の API は、以下の原則に基づいて設計されています:

1. **直感的**: PyTorch や NumPy に慣れたユーザーが即座に理解できる
2. **一貫性**: 命名規則とパターンが統一されている
3. **型安全**: C++ のテンプレートと型システムを活用
4. **Pythonic**: Python API は Python らしい記法を提供
5. **ゼロコスト抽象化**: 抽象化によるオーバーヘッドを最小化

---

## C++ API

### Tensor 作成

#### 基本的な作成方法

```cpp
using namespace fullscratch::autograd;

// 形状を指定して作成
Tensor<float> t1({2, 3});  // 未初期化

// データから作成
std::vector<float> data = {1, 2, 3, 4, 5, 6};
Tensor<float> t2({2, 3}, data);

// 初期化リストから作成
Tensor<float> t3 = {{1, 2, 3}, {4, 5, 6}};

// デバイスを指定
Tensor<float> t4({2, 3}, DeviceType::CUDA);
```

#### ファクトリ関数

```cpp
// ゼロ初期化
auto zeros = Tensor<float>::zeros({3, 4});

// 1 で初期化
auto ones = Tensor<float>::ones({3, 4});

// 正規分布からサンプリング
auto randn = Tensor<float>::randn({3, 4}, /*mean=*/0.0, /*std=*/1.0);

// 一様分布からサンプリング
auto rand = Tensor<float>::rand({3, 4}, /*low=*/0.0, /*high=*/1.0);

// 等差数列
auto arange = Tensor<float>::arange(0, 10, 1);  // [0, 1, 2, ..., 9]

// 等間隔の数列
auto linspace = Tensor<float>::linspace(0, 1, 11);  // [0, 0.1, 0.2, ..., 1.0]

// 単位行列
auto eye = Tensor<float>::eye(3);

// 別のテンソルと同じ形状
auto zeros_like = Tensor<float>::zeros_like(t1);
auto ones_like = Tensor<float>::ones_like(t1);
```

### Tensor 操作

#### 形状変換

```cpp
auto t = Tensor<float>::randn({2, 3, 4});

// 形状の取得
auto shape = t.shape();           // Shape([2, 3, 4])
size_t ndim = t.ndim();           // 3
size_t size = t.size();           // 24

// Reshape
auto t1 = t.reshape({6, 4});      // [6, 4]
auto t2 = t.view({2, 12});        // [2, 12] (ゼロコピー)
auto t3 = t.flatten();            // [24]

// Transpose
auto t4 = t.transpose(0, 1);      // [3, 2, 4]
auto t5 = t.permute({2, 0, 1});   // [4, 2, 3]

// Squeeze / Unsqueeze
auto t6 = Tensor<float>::randn({1, 3, 1, 4});
auto t7 = t6.squeeze();           // [3, 4]
auto t8 = t7.unsqueeze(1);        // [3, 1, 4]

// 連続メモリへの変換
bool is_cont = t.is_contiguous();
auto cont = t.contiguous();
```

#### インデックスとスライシング

```cpp
auto t = Tensor<float>::randn({4, 5, 6});

// 単一要素へのアクセス
float val = t[{1, 2, 3}];
t[{1, 2, 3}] = 42.0f;

// スライシング
auto t1 = t.slice(0, 1, 3);       // dim=0 で [1, 3) を取得 → [2, 5, 6]
auto t2 = t.slice(1, 0, 3);       // dim=1 で [0, 3) を取得 → [4, 3, 6]

// 複数次元のスライシング
auto t3 = t[Slice(0, 2)][Slice(1, 4)][Slice()];  // [2, 3, 6]

// マスキング
auto mask = t > 0;
auto positive = t.masked_select(mask);

// インデックス選択
auto indices = Tensor<int64_t>({2}, {0, 2});
auto selected = t.index_select(0, indices);  // [2, 5, 6]
```

#### デバイス間の移動

```cpp
auto cpu_tensor = Tensor<float>::randn({3, 4});

// GPU に転送
auto gpu_tensor = cpu_tensor.to(DeviceType::CUDA);

// CPU に戻す
auto back_to_cpu = gpu_tensor.to(DeviceType::CPU);

// デバイスを指定して新しいテンソルを作成
auto cuda_tensor = Tensor<float>::zeros({3, 4}, DeviceType::CUDA);

// デバイスの確認
DeviceType device = cpu_tensor.device();  // DeviceType::CPU
```

### Variable と自動微分

#### Variable の作成

```cpp
// Tensor から Variable を作成
auto t = Tensor<float>::randn({3, 4});
auto v = Variable<float>(t, /*requires_grad=*/true);

// ファクトリ関数でも作成可能
auto v2 = Variable<float>::randn({3, 4}, /*requires_grad=*/true);
```

#### Forward と Backward

```cpp
// モデルの定義
auto x = Variable<float>::randn({64, 10}, true);  // 入力
auto W1 = Variable<float>::randn({10, 20}, true);
auto b1 = Variable<float>::zeros({20}, true);
auto W2 = Variable<float>::randn({20, 5}, true);
auto b2 = Variable<float>::zeros({5}, true);

// Forward pass
auto h = relu(matmul(x, W1) + b1);
auto y = matmul(h, W2) + b2;

// 損失の計算
auto target = Variable<float>::randn({64, 5}, false);
auto loss = mse_loss(y, target);

// Backward pass
loss.backward();

// 勾配の取得
auto grad_W1 = W1.grad();
auto grad_b1 = b1.grad();
auto grad_W2 = W2.grad();
auto grad_b2 = b2.grad();

// 勾配のゼロクリア
W1.zero_grad();
b1.zero_grad();
W2.zero_grad();
b2.zero_grad();
```

#### no_grad コンテキスト

```cpp
// 勾配計算を無効化（推論時など）
{
    NoGradGuard no_grad;

    auto y_pred = model.forward(x);
    // ここでは計算グラフが構築されない
}
```

### 演算

#### 基本的な算術演算

```cpp
auto a = Variable<float>::randn({3, 4}, true);
auto b = Variable<float>::randn({3, 4}, true);

// 要素ごとの演算
auto c1 = a + b;
auto c2 = a - b;
auto c3 = a * b;
auto c4 = a / b;
auto c5 = a.pow(2);

// スカラー演算
auto c6 = a + 1.0f;
auto c7 = a * 2.0f;

// Inplace 演算（勾配計算が不要な場合）
a.add_(1.0f);
a.mul_(2.0f);

// 行列演算
auto A = Variable<float>::randn({3, 4}, true);
auto B = Variable<float>::randn({4, 5}, true);
auto C = matmul(A, B);  // [3, 5]

// Broadcasting
auto x = Variable<float>::randn({3, 1}, true);
auto y = Variable<float>::randn({4}, true);
auto z = x + y;  // [3, 4]
```

#### 活性化関数

```cpp
auto x = Variable<float>::randn({64, 100}, true);

auto y1 = relu(x);
auto y2 = sigmoid(x);
auto y3 = tanh(x);
auto y4 = gelu(x);
auto y5 = leaky_relu(x, /*negative_slope=*/0.01);
auto y6 = elu(x, /*alpha=*/1.0);
auto y7 = silu(x);  // Swish
auto y8 = softplus(x);
```

#### Softmax と LogSoftmax

```cpp
auto logits = Variable<float>::randn({64, 10}, true);

auto probs = softmax(logits, /*dim=*/-1);
auto log_probs = log_softmax(logits, /*dim=*/-1);
```

#### 損失関数

```cpp
auto y_pred = Variable<float>::randn({64, 10}, true);
auto y_true = Variable<float>::randn({64, 10}, false);

// Mean Squared Error
auto loss1 = mse_loss(y_pred, y_true);

// Cross Entropy Loss
auto logits = Variable<float>::randn({64, 10}, true);
auto labels = Tensor<int64_t>::randint(0, 10, {64});
auto loss2 = cross_entropy_loss(logits, labels);

// Binary Cross Entropy
auto probs = sigmoid(y_pred);
auto binary_labels = Variable<float>::randint(0, 2, {64, 10}, false);
auto loss3 = binary_cross_entropy_loss(probs, binary_labels);

// Reduction の指定
auto loss4 = mse_loss(y_pred, y_true, Reduction::MEAN);  // 平均
auto loss5 = mse_loss(y_pred, y_true, Reduction::SUM);   // 合計
auto loss6 = mse_loss(y_pred, y_true, Reduction::NONE);  // 要素ごと
```

#### 集約演算

```cpp
auto t = Variable<float>::randn({3, 4, 5}, true);

// Sum
auto sum_all = t.sum();                    // スカラー
auto sum_dim0 = t.sum(0);                  // [4, 5]
auto sum_dim01 = t.sum({0, 1});            // [5]
auto sum_keepdim = t.sum(0, /*keepdim=*/true);  // [1, 4, 5]

// Mean
auto mean = t.mean();
auto mean_dim = t.mean(1);

// Max / Min
auto max_val = t.max();
auto max_dim = t.max(0);  // (values, indices) のタプルを返す
auto [values, indices] = t.max(0);

auto min_val = t.min();
auto min_dim = t.min(0);

// Variance / Std
auto var = t.var();
auto std = t.std();
auto var_unbiased = t.var(/*unbiased=*/true);
```

### Optimizer

#### SGD

```cpp
#include <fullscratch/optim/sgd.hpp>

// パラメータのリスト
std::vector<Variable<float>*> params = {&W1, &b1, &W2, &b2};

// Optimizer の作成
SGD<float> optimizer(params, /*lr=*/0.01, /*momentum=*/0.9);

// 学習ループ
for (int epoch = 0; epoch < 100; ++epoch) {
    // Forward + Backward
    auto loss = compute_loss();
    loss.backward();

    // パラメータ更新
    optimizer.step();

    // 勾配をゼロクリア
    optimizer.zero_grad();
}
```

#### Adam

```cpp
#include <fullscratch/optim/adam.hpp>

Adam<float> optimizer(params,
    /*lr=*/0.001,
    /*beta1=*/0.9,
    /*beta2=*/0.999,
    /*eps=*/1e-8,
    /*weight_decay=*/0.0
);

for (int epoch = 0; epoch < 100; ++epoch) {
    auto loss = compute_loss();
    loss.backward();
    optimizer.step();
    optimizer.zero_grad();
}
```

#### Learning Rate Scheduler

```cpp
#include <fullscratch/optim/lr_scheduler.hpp>

// StepLR: 一定エポックごとに学習率を減衰
StepLR<float> scheduler(optimizer, /*step_size=*/30, /*gamma=*/0.1);

// CosineAnnealingLR: コサイン減衰
CosineAnnealingLR<float> scheduler(optimizer, /*T_max=*/100);

for (int epoch = 0; epoch < 100; ++epoch) {
    // Training
    train_one_epoch();

    // 学習率を更新
    scheduler.step();
}
```

### Transformer コンポーネント

#### MultiHeadAttention

```cpp
#include <fullscratch/nn/attention.hpp>

// MultiHeadAttention の作成
MultiHeadAttention<float> attn(
    /*d_model=*/512,
    /*num_heads=*/8,
    /*dropout=*/0.1
);

// Forward pass
auto query = Variable<float>::randn({32, 10, 512}, true);  // [batch, seq_len, d_model]
auto key = Variable<float>::randn({32, 10, 512}, true);
auto value = Variable<float>::randn({32, 10, 512}, true);

// Mask (optional)
auto mask = Tensor<float>::ones({10, 10});  // [seq_len, seq_len]
// Causal mask for decoder
for (size_t i = 0; i < 10; ++i) {
    for (size_t j = i + 1; j < 10; ++j) {
        mask[{i, j}] = 0;
    }
}

auto output = attn.forward(query, key, value, &Variable<float>(mask, false));
```

#### TransformerEncoderLayer

```cpp
#include <fullscratch/nn/transformer.hpp>

TransformerEncoderLayer<float> encoder_layer(
    /*d_model=*/512,
    /*num_heads=*/8,
    /*d_ff=*/2048,
    /*dropout=*/0.1
);

auto x = Variable<float>::randn({32, 10, 512}, true);
auto output = encoder_layer.forward(x);
```

#### TransformerEncoder

```cpp
TransformerEncoder<float> encoder(
    /*num_layers=*/6,
    /*d_model=*/512,
    /*num_heads=*/8,
    /*d_ff=*/2048,
    /*dropout=*/0.1
);

auto x = Variable<float>::randn({32, 10, 512}, true);
auto encoded = encoder.forward(x);
```

#### TransformerDecoderLayer

```cpp
TransformerDecoderLayer<float> decoder_layer(
    /*d_model=*/512,
    /*num_heads=*/8,
    /*d_ff=*/2048,
    /*dropout=*/0.1
);

auto tgt = Variable<float>::randn({32, 20, 512}, true);  // Target sequence
auto memory = Variable<float>::randn({32, 10, 512}, true);  // Encoder output

auto output = decoder_layer.forward(tgt, memory, /*tgt_mask=*/nullptr, /*memory_mask=*/nullptr);
```

#### Transformer (Full Model)

```cpp
Transformer<float> transformer(
    /*d_model=*/512,
    /*num_heads=*/8,
    /*num_encoder_layers=*/6,
    /*num_decoder_layers=*/6,
    /*d_ff=*/2048,
    /*dropout=*/0.1,
    /*max_seq_len=*/5000
);

auto src = Variable<float>::randn({32, 10, 512}, true);
auto tgt = Variable<float>::randn({32, 20, 512}, true);

auto output = transformer.forward(src, tgt);  // [32, 20, 512]
```

---

## Python API

Python API は nanobind を使用して C++ の実装をラップします。nanobind は pybind11 の後継で、より高速でコンパクトなバインディングを提供します。API は PyTorch に極めて近い形で提供されます。

### インポート

```python
import gradflow as gf
import gradflow.nn as nn
import gradflow.optim as optim
```

### Tensor 作成

```python
# NumPy 配列から作成
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
t = gf.tensor(a)

# リストから作成
t = gf.tensor([[1, 2, 3], [4, 5, 6]], dtype=gf.float32)

# ファクトリ関数
zeros = gf.zeros((3, 4))
ones = gf.ones((3, 4))
randn = gf.randn((3, 4))
rand = gf.rand((3, 4))
arange = gf.arange(0, 10, 1)
linspace = gf.linspace(0, 1, 11)
eye = gf.eye(3)

# デバイスを指定
metal_tensor = gf.zeros((3, 4), device='metal')  # Apple Silicon
cuda_tensor = gf.zeros((3, 4), device='cuda')    # NVIDIA GPU
cpu_tensor = gf.zeros((3, 4), device='cpu')
```

### Tensor 操作

```python
t = gf.randn((2, 3, 4))

# 形状の取得
shape = t.shape  # (2, 3, 4)
ndim = t.ndim    # 3
size = t.size()  # 24

# Reshape
t1 = t.reshape((6, 4))
t2 = t.view((2, 12))
t3 = t.flatten()

# Transpose
t4 = t.transpose(0, 1)
t5 = t.permute((2, 0, 1))

# インデックス（NumPy と互換）
val = t[0, 1, 2]
t[0, 1, 2] = 42.0

slice_t = t[0:2, 1:3, :]

# デバイス移動
gpu_t = t.cuda()
cpu_t = gpu_t.cpu()
# または
gpu_t = t.to('cuda')
cpu_t = t.to('cpu')

# NumPy への変換
np_array = t.numpy()
```

### Variable と自動微分

```python
# requires_grad を指定
x = gf.randn((64, 10), requires_grad=True)
W = gf.randn((10, 5), requires_grad=True)
b = gf.zeros((5,), requires_grad=True)

# Forward
y = gf.matmul(x, W) + b

# Backward
y.backward(gf.ones_like(y))

# 勾配の取得
grad_x = x.grad
grad_W = W.grad
grad_b = b.grad

# 勾配をゼロクリア
x.zero_grad()
W.zero_grad()
b.zero_grad()

# no_grad コンテキスト
with gf.no_grad():
    y_pred = model(x)
```

### nn.Module による層の定義

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 784)
        x = gf.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = MyModel()

# パラメータの取得
params = model.parameters()

# GPU に転送
model = model.cuda()
```

### Transformer の使用

```python
import gradflow.nn as nn

# MultiHeadAttention
attn = nn.MultiHeadAttention(d_model=512, num_heads=8, dropout=0.1)

query = gf.randn((32, 10, 512))
key = gf.randn((32, 10, 512))
value = gf.randn((32, 10, 512))

output = attn(query, key, value)

# TransformerEncoder
encoder = nn.TransformerEncoder(
    num_layers=6,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    dropout=0.1
)

x = gf.randn((32, 10, 512))
encoded = encoder(x)

# Full Transformer
transformer = nn.Transformer(
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048,
    dropout=0.1
)

src = gf.randn((32, 10, 512))
tgt = gf.randn((32, 20, 512))

output = transformer(src, tgt)
```

### Optimizer

```python
import gradflow.optim as optim

# SGD
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# AdamW
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# 学習ループ
for epoch in range(100):
    # Forward
    output = model(input)
    loss = gf.cross_entropy_loss(output, target)

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Update
    optimizer.step()
```

### Learning Rate Scheduler

```python
from fullscratchml.optim.lr_scheduler import StepLR, CosineAnnealingLR

# StepLR
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    train_one_epoch()
    scheduler.step()
```

---

## 使用例

### 例 1: 簡単な線形回帰

#### C++

```cpp
#include <fullscratch/autograd.hpp>
#include <fullscratch/optim/sgd.hpp>

int main() {
    using namespace fullscratch::autograd;

    // データの生成: y = 2x + 3 + noise
    auto X = Variable<float>::randn({100, 1}, false);
    auto y_true = Variable<float>(X.data() * 2.0f + 3.0f, false);

    // パラメータ
    auto W = Variable<float>::randn({1, 1}, true);
    auto b = Variable<float>::zeros({1}, true);

    // Optimizer
    std::vector<Variable<float>*> params = {&W, &b};
    SGD<float> optimizer(params, 0.01);

    // 学習
    for (int epoch = 0; epoch < 1000; ++epoch) {
        // Forward
        auto y_pred = matmul(X, W) + b;
        auto loss = mse_loss(y_pred, y_true);

        // Backward
        loss.backward();

        // Update
        optimizer.step();
        optimizer.zero_grad();

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss.data()[{0}] << std::endl;
        }
    }

    std::cout << "W: " << W.data()[{0, 0}] << ", b: " << b.data()[{0}] << std::endl;

    return 0;
}
```

#### Python

```python
import gradflow as fs
import gradflow.optim as optim

# データの生成
X = gf.randn((100, 1))
y_true = X * 2.0 + 3.0

# パラメータ
W = gf.randn((1, 1), requires_grad=True)
b = gf.zeros((1,), requires_grad=True)

# Optimizer
optimizer = optim.SGD([W, b], lr=0.01)

# 学習
for epoch in range(1000):
    # Forward
    y_pred = gf.matmul(X, W) + b
    loss = gf.mse_loss(y_pred, y_true)

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Update
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print(f"W: {W.item()}, b: {b.item()}")
```

### 例 2: MNIST 分類

#### C++

```cpp
#include <fullscratch/fullscratch.hpp>

class MNISTModel {
public:
    MNISTModel() {
        W1_ = Variable<float>::randn({784, 128}, true);
        b1_ = Variable<float>::zeros({128}, true);
        W2_ = Variable<float>::randn({128, 10}, true);
        b2_ = Variable<float>::zeros({10}, true);
    }

    Variable<float> forward(const Variable<float>& x) {
        auto h = relu(matmul(x, W1_) + b1_);
        auto y = matmul(h, W2_) + b2_;
        return y;
    }

    std::vector<Variable<float>*> parameters() {
        return {&W1_, &b1_, &W2_, &b2_};
    }

private:
    Variable<float> W1_, b1_, W2_, b2_;
};

int main() {
    // モデルの作成
    MNISTModel model;

    // Optimizer
    Adam<float> optimizer(model.parameters(), 0.001);

    // 学習ループ
    for (int epoch = 0; epoch < 10; ++epoch) {
        for (auto& [x_batch, y_batch] : train_loader) {
            // Forward
            auto logits = model.forward(x_batch);
            auto loss = cross_entropy_loss(logits, y_batch);

            // Backward
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        // 評価
        float accuracy = evaluate(model, test_loader);
        std::cout << "Epoch " << epoch << ", Accuracy: " << accuracy << std::endl;
    }

    return 0;
}
```

#### Python

```python
import gradflow as fs
import gradflow.nn as nn
import gradflow.optim as optim

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = gf.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# モデルの作成
model = MNISTModel()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習ループ
for epoch in range(10):
    for x_batch, y_batch in train_loader:
        # Forward
        logits = model(x_batch)
        loss = gf.cross_entropy_loss(logits, y_batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 評価
    accuracy = evaluate(model, test_loader)
    print(f"Epoch {epoch}, Accuracy: {accuracy}")
```

### 例 3: Transformer による言語モデル

#### Python

```python
import gradflow as fs
import gradflow.nn as nn

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.PositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        # x: [batch, seq_len]
        x = self.embedding(x)  # [batch, seq_len, d_model]
        x = self.pos_encoding(x)
        x = self.transformer(x, mask=mask)
        logits = self.fc_out(x)  # [batch, seq_len, vocab_size]
        return logits

# モデルの作成
model = TransformerLM(vocab_size=10000)

# 学習
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(100):
    for batch in data_loader:
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']

        # Forward
        logits = model(input_ids)

        # Loss (各トークンの予測)
        loss = gf.cross_entropy_loss(
            logits.view(-1, 10000),
            target_ids.view(-1)
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## API リファレンス

### Tensor Methods

| メソッド | 説明 | 戻り値 |
|---------|------|--------|
| `shape()` | テンソルの形状を取得 | `Shape` |
| `ndim()` | 次元数を取得 | `size_t` |
| `size()` | 総要素数を取得 | `size_t` |
| `device()` | デバイスタイプを取得 | `DeviceType` |
| `dtype()` | データ型を取得 | `DType` |
| `reshape(shape)` | 形状を変更 | `Tensor` |
| `view(shape)` | 形状を変更（ゼロコピー） | `Tensor` |
| `transpose(dim0, dim1)` | 2次元を入れ替え | `Tensor` |
| `permute(dims)` | 次元を並び替え | `Tensor` |
| `squeeze(dim=-1)` | サイズ1の次元を削除 | `Tensor` |
| `unsqueeze(dim)` | 次元を追加 | `Tensor` |
| `flatten()` | 1次元に平坦化 | `Tensor` |
| `slice(dim, start, end)` | スライス | `Tensor` |
| `to(device)` | デバイスに転送 | `Tensor` |
| `contiguous()` | 連続メモリに変換 | `Tensor` |
| `is_contiguous()` | 連続メモリか判定 | `bool` |

### Variable Methods

| メソッド | 説明 | 戻り値 |
|---------|------|--------|
| `data()` | データテンソルを取得 | `Tensor&` |
| `grad()` | 勾配テンソルを取得 | `Tensor&` |
| `requires_grad()` | 勾配計算の要否を取得 | `bool` |
| `backward(grad)` | 逆伝播を実行 | `void` |
| `zero_grad()` | 勾配をゼロクリア | `void` |

### 演算関数

| 関数 | 説明 |
|------|------|
| `matmul(a, b)` | 行列積 |
| `add(a, b)` | 加算 |
| `sub(a, b)` | 減算 |
| `mul(a, b)` | 乗算 |
| `div(a, b)` | 除算 |
| `pow(a, exponent)` | べき乗 |
| `relu(x)` | ReLU 活性化 |
| `sigmoid(x)` | Sigmoid 活性化 |
| `tanh(x)` | Tanh 活性化 |
| `gelu(x)` | GELU 活性化 |
| `softmax(x, dim)` | Softmax |
| `log_softmax(x, dim)` | Log Softmax |
| `layer_norm(x, normalized_shape)` | Layer Normalization |

### 損失関数

| 関数 | 説明 |
|------|------|
| `mse_loss(pred, target)` | Mean Squared Error |
| `cross_entropy_loss(logits, labels)` | Cross Entropy |
| `binary_cross_entropy_loss(probs, labels)` | Binary Cross Entropy |
| `nll_loss(log_probs, labels)` | Negative Log Likelihood |

---

## まとめ

本 API は、以下の特徴を持ちます:

1. **C++ と Python の統一**: 同じ概念が両方の言語で一貫して表現される
2. **PyTorch との互換性**: PyTorch ユーザーが即座に使い始められる
3. **型安全性**: C++ テンプレートによる強力な型チェック
4. **ゼロコスト抽象化**: 抽象化によるパフォーマンスペナルティを最小化

次のステップは、[実装ロードマップ](./ROADMAP.md) を参照してください。
