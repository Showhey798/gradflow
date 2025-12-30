# Issue #10: 活性化関数の実装 - 前提条件と準備

## 1. 概要

Issue #10 の実装を開始する前に、必要な Tensor レベルの演算が実装されているかを確認します。不足している演算は Phase 1 で追加する必要があります。

## 2. 必要な Tensor レベルの演算

### 2.1 既存の演算（確認済み）

以下の演算は既に実装されています（Issue #9 で使用）：

| 演算 | ファイル | 状態 |
|------|---------|------|
| `add(a, b)` | `elementwise.hpp` | ✅ 実装済み |
| `sub(a, b)` | `elementwise.hpp` | ✅ 実装済み |
| `mul(a, b)` | `elementwise.hpp` | ✅ 実装済み |
| `div(a, b)` | `elementwise.hpp` | ✅ 実装済み |
| `exp(a)` | `elementwise.hpp` | ✅ 実装済み |
| `log(a)` | `elementwise.hpp` | ✅ 実装済み |
| `sqrt(a)` | `elementwise.hpp` | ✅ 実装済み |
| `pow(a, b)` | `elementwise.hpp` | ✅ 実装済み |
| `sum(a)` | `reduction.hpp` | ✅ 実装済み |
| `sum(a, axis)` | `reduction.hpp` | ✅ 実装済み |

### 2.2 確認が必要な演算

以下の演算が実装されているかを確認する必要があります：

| 演算 | 必要な活性化関数 | 優先度 | 状態 |
|------|-----------------|--------|------|
| `tanh(a)` | Tanh, GELU | 高 | ⏳ 確認が必要 |
| `max(a, scalar)` | ReLU | 高 | ⏳ 確認が必要 |
| `max(a, axis, keepdim)` | Softmax, LogSoftmax | 中 | ⏳ 確認が必要 |
| `sum(a, axis, keepdim)` | Softmax, LogSoftmax | 中 | ⏳ 確認が必要 |

### 2.3 確認手順

以下のコマンドで演算の実装を確認します：

```bash
# tanh の実装を確認
grep -r "tanh" include/gradflow/

# max の実装を確認
grep -r "max" include/gradflow/autograd/ops/

# sum の keepdim オプションを確認
grep -A 10 "sum.*axis" include/gradflow/autograd/ops/reduction.hpp
```

## 3. 各演算の実装方針

### 3.1 `tanh(a)` - 双曲線正接

**必要な活性化関数**: Tanh, GELU

**実装方針**:
- `elementwise.hpp` に追加
- 標準ライブラリの `std::tanh` を使用

**実装例**:
```cpp
template <typename T>
Tensor<T> tanh(const Tensor<T>& a) {
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result.data()[i] = std::tanh(a.data()[i]);
    }
    return result;
}
```

**テスト**:
```cpp
TEST(ElementwiseTest, Tanh) {
    auto x = Tensor<float>({-1.0F, 0.0F, 1.0F});
    auto result = tanh(x);

    EXPECT_NEAR(result[{0}], -0.7616F, 1e-4F);  // tanh(-1)
    EXPECT_NEAR(result[{1}], 0.0F, 1e-5F);      // tanh(0)
    EXPECT_NEAR(result[{2}], 0.7616F, 1e-4F);   // tanh(1)
}
```

---

### 3.2 `max(a, scalar)` - スカラーとの最大値

**必要な活性化関数**: ReLU

**実装方針**:
- `elementwise.hpp` に追加
- または ReLU の forward で直接実装

**実装例 1（Tensor レベル）**:
```cpp
template <typename T>
Tensor<T> max(const Tensor<T>& a, T scalar) {
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.size(); ++i) {
        result.data()[i] = std::max(a.data()[i], scalar);
    }
    return result;
}
```

**実装例 2（ReLU の forward で直接実装）**:
```cpp
// relu.hpp の forward 内
Tensor<T> result(x.shape());
for (size_t i = 0; i < x.size(); ++i) {
    result.data()[i] = std::max(x.data()[i], T(0));
}
```

**推奨**: 実装例 2（ReLU 内で直接実装）の方がシンプルです。

---

### 3.3 `max(a, axis, keepdim)` - 指定軸の最大値

**必要な活性化関数**: Softmax, LogSoftmax

**実装方針**:
- `reduction.hpp` に追加
- `sum(a, axis)` の実装を参考に

**実装例**:
```cpp
template <typename T>
Tensor<T> max(const Tensor<T>& a, size_t axis, bool keepdim = false) {
    if (axis >= a.ndim()) {
        throw std::out_of_range("Axis out of range");
    }

    // Compute output shape
    std::vector<size_t> result_dims;
    for (size_t i = 0; i < a.ndim(); ++i) {
        if (i != axis) {
            result_dims.push_back(a.shape()[i]);
        } else if (keepdim) {
            result_dims.push_back(1);
        }
    }

    Shape result_shape(result_dims.empty() ? std::vector<size_t>{} : result_dims);
    Tensor<T> result(result_shape);

    // Initialize result to minimum value
    detail::iterateIndices(result_shape, [&](const std::vector<size_t>& indices) {
        result[indices] = std::numeric_limits<T>::lowest();
    });

    // Find max over the specified axis
    detail::iterateIndices(a.shape(), [&](const std::vector<size_t>& a_indices) {
        std::vector<size_t> result_indices;
        for (size_t i = 0; i < a.ndim(); ++i) {
            if (i != axis) {
                result_indices.push_back(a_indices[i]);
            } else if (keepdim) {
                result_indices.push_back(0);
            }
        }
        result[result_indices] = std::max(result[result_indices], a[a_indices]);
    });

    return result;
}
```

**テスト**:
```cpp
TEST(ReductionTest, MaxWithKeepdim) {
    auto x = Tensor<float>({{1.0F, 3.0F, 2.0F},
                            {4.0F, 2.0F, 5.0F}});

    // max along axis 1, keepdim=true
    auto result = max(x, 1, true);

    EXPECT_EQ(result.shape(), Shape({2, 1}));
    EXPECT_FLOAT_EQ((result[{0, 0}]), 3.0F);
    EXPECT_FLOAT_EQ((result[{1, 0}]), 5.0F);
}
```

---

### 3.4 `sum(a, axis, keepdim)` - 指定軸の総和（keepdim オプション）

**必要な活性化関数**: Softmax, LogSoftmax

**実装方針**:
- `reduction.hpp` の既存の `sum(a, axis)` を拡張
- `keepdim` パラメータを追加

**実装例**:
```cpp
template <typename T>
Tensor<T> sum(const Tensor<T>& a, size_t axis, bool keepdim = false) {
    if (axis >= a.ndim()) {
        throw std::out_of_range("Axis out of range");
    }

    // Compute output shape
    std::vector<size_t> result_dims;
    for (size_t i = 0; i < a.ndim(); ++i) {
        if (i != axis) {
            result_dims.push_back(a.shape()[i]);
        } else if (keepdim) {
            result_dims.push_back(1);
        }
    }

    Shape result_shape(result_dims.empty() ? std::vector<size_t>{} : result_dims);
    Tensor<T> result(result_shape);

    // Initialize result to zero
    detail::iterateIndices(result_shape,
                           [&](const std::vector<size_t>& indices) { result[indices] = T(0); });

    // Sum over the specified axis
    detail::iterateIndices(a.shape(), [&](const std::vector<size_t>& a_indices) {
        std::vector<size_t> result_indices;
        for (size_t i = 0; i < a.ndim(); ++i) {
            if (i != axis) {
                result_indices.push_back(a_indices[i]);
            } else if (keepdim) {
                result_indices.push_back(0);
            }
        }
        result[result_indices] += a[a_indices];
    });

    return result;
}
```

**注意**: 既存の `sum(a, axis)` が `keepdim=false` の動作をしている場合は、デフォルト引数を追加するだけで拡張できます。

---

## 4. 実装の優先順位

### Phase 0: 前提条件の確認と実装（1-2 日）

**優先度: 高**

1. **`tanh(a)` の実装**
   - ファイル: `include/gradflow/autograd/ops/elementwise.hpp`
   - テスト: `tests/test_elementwise.cpp` に追加
   - 影響する活性化関数: Tanh, GELU

2. **`max(a, axis, keepdim)` の実装**
   - ファイル: `include/gradflow/autograd/ops/reduction.hpp`
   - テスト: `tests/test_reduction.cpp` に追加
   - 影響する活性化関数: Softmax, LogSoftmax

3. **`sum(a, axis, keepdim)` の拡張**
   - ファイル: `include/gradflow/autograd/ops/reduction.hpp`
   - テスト: `tests/test_reduction.cpp` に追加
   - 影響する活性化関数: Softmax, LogSoftmax

**完了基準**:
- ✅ `tanh` のテストが pass
- ✅ `max(a, axis, keepdim)` のテストが pass
- ✅ `sum(a, axis, keepdim)` のテストが pass

### Phase 1: 基本的な活性化関数（1-3 日）

Phase 0 が完了してから開始します。

---

## 5. 代替案

Softmax/LogSoftmax の実装で `keepdim` オプションが不要な場合、以下の代替案があります：

### 代替案 1: `reshape` で次元を追加

```cpp
// max を計算（keepdim なし）
auto max_x = max(x, actual_dim);  // [2, 3] -> [2]

// reshape で次元を追加
auto max_x_keepdim = max_x.reshape(Shape({2, 1}));  // [2] -> [2, 1]
```

**前提**: `reshape` が実装されている必要があります。

### 代替案 2: Broadcasting を利用

```cpp
// max を計算（keepdim なし）
auto max_x = max(x, actual_dim);  // [2, 3] -> [2]

// Broadcasting で次元を拡張
auto x_shifted = sub(x, max_x);  // Broadcasting: [2, 3] - [2] -> [2, 3]
```

**前提**: Broadcasting が正しく動作する必要があります。

---

## 6. チェックリスト

Phase 0 を開始する前に、以下をチェックしてください：

- [ ] `tanh` が実装されているか確認
  - `grep -r "tanh" include/gradflow/`
  - 実装されていない → Phase 0 で実装

- [ ] `max(a, axis)` が実装されているか確認
  - `grep -r "max" include/gradflow/autograd/ops/reduction.hpp`
  - 実装されていない → Phase 0 で実装

- [ ] `sum(a, axis, keepdim)` が実装されているか確認
  - `grep -A 5 "sum.*axis" include/gradflow/autograd/ops/reduction.hpp`
  - `keepdim` オプションがない → Phase 0 で拡張

- [ ] `reshape` が実装されているか確認（代替案用）
  - `grep -r "reshape" include/gradflow/`

---

## 7. まとめ

Issue #10 の実装を開始する前に、以下の 3 つの演算が必要です：

1. **`tanh(a)`**: 優先度高（Tanh, GELU で使用）
2. **`max(a, axis, keepdim)`**: 優先度中（Softmax, LogSoftmax で使用）
3. **`sum(a, axis, keepdim)`**: 優先度中（Softmax, LogSoftmax で使用）

これらの演算が実装されているかを確認し、不足している場合は Phase 0 で追加します。Phase 0 が完了してから、Phase 1（基本的な活性化関数）に進みます。
