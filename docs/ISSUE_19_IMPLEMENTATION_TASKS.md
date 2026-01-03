# Issue #19: CPU 最適化実装タスクリスト

このドキュメントは、github-issue-implementer が Issue #19 の CPU 最適化を実装するための詳細なタスクリストです。各タスクは独立して実装可能で、テスト駆動開発（TDD）のサイクルに従います。

---

## 実装の前提条件

### 依存 Issue

- ✅ Issue #18 (Phase 3 統合テスト) - 完了

### 必要なツール・ライブラリ

- C++17 コンパイラ（GCC 11+, Clang 14+, または MSVC 2019+）
- AVX2 サポート（x86_64 プロセッサ）
- OpenMP ライブラリ
- Google Test（テストフレームワーク）
- Eigen3（オプション、ベンチマーク比較用）

---

## Phase 1: プロジェクト構造のセットアップ

### Task 1.1: ディレクトリ構造の作成

**説明**: CPU 最適化のためのディレクトリ構造を作成します。

**作業内容**:

1. 以下のディレクトリを作成
   ```
   src/autograd/cpu/
   include/gradflow/autograd/cpu/
   ```

2. ディレクトリ構造を確認
   ```bash
   ls -la src/autograd/cpu/
   ls -la include/gradflow/autograd/cpu/
   ```

**完了基準**:
- [ ] ディレクトリが作成されている

**推定時間**: 5 分

---

### Task 1.2: CMake ファイルの作成

**説明**: CPU 最適化のための CMake 設定を追加します。

**作業内容**:

1. `src/autograd/cpu/CMakeLists.txt` を作成

```cmake
# src/autograd/cpu/CMakeLists.txt

# CPU 最適化カーネルの設定
if(GRADFLOW_ENABLE_CPU_OPTIMIZATIONS)
    message(STATUS "Enabling CPU optimizations")

    # AVX2 フラグ
    if(MSVC)
        set(AVX2_FLAGS /arch:AVX2)
    else()
        set(AVX2_FLAGS -mavx2 -mfma)
    endif()

    # OpenMP サポート
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    else()
        message(WARNING "OpenMP not found, parallel MatMul will be disabled")
    endif()

    # CPU 最適化カーネルのソース
    set(CPU_OPT_SOURCES
        cpu/kernels_avx2.cpp
        cpu/matmul_blocked.cpp
    )

    # gradflow_impl ライブラリに CPU 最適化ソースを追加
    target_sources(gradflow_impl PRIVATE ${CPU_OPT_SOURCES})

    # AVX2 コンパイルフラグを追加
    target_compile_options(gradflow_impl PRIVATE ${AVX2_FLAGS})

    # OpenMP をリンク
    if(OpenMP_CXX_FOUND)
        target_link_libraries(gradflow_impl PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()
```

2. ルート `CMakeLists.txt` に以下を追加（適切な位置に挿入）

```cmake
# Build Options セクションに追加
option(GRADFLOW_ENABLE_CPU_OPTIMIZATIONS "Enable CPU optimizations (SIMD, OpenMP)" ON)
```

3. `src/CMakeLists.txt` に以下を追加

```cmake
# CPU 最適化サブディレクトリを追加
if(GRADFLOW_ENABLE_CPU_OPTIMIZATIONS)
    add_subdirectory(autograd/cpu)
endif()
```

**完了基準**:
- [ ] CMake ファイルが作成されている
- [ ] `cmake -DGRADFLOW_ENABLE_CPU_OPTIMIZATIONS=ON ..` が成功する

**推定時間**: 15 分

---

## Phase 2: SIMD Intrinsic 演算の実装

### Task 2.1: SIMD Ops ヘッダーファイルの作成

**説明**: AVX2 intrinsic を使用した低レベル SIMD 演算を実装します。

**作業内容**:

1. `src/autograd/cpu/simd_ops.hpp` を作成

```cpp
// src/autograd/cpu/simd_ops.hpp
#pragma once

#include <immintrin.h>  // AVX2/AVX512 intrinsics
#include <cstddef>

namespace gradflow {
namespace cpu {
namespace simd {

/**
 * @brief AVX2 を使用した要素ごとの加算
 *
 * 32 バイトアライメント必須。サイズは 8 の倍数であることを推奨。
 *
 * @param a 入力配列 A (32-byte aligned)
 * @param b 入力配列 B (32-byte aligned)
 * @param c 出力配列 C (32-byte aligned)
 * @param size 要素数
 */
inline void add_avx2(const float* a, const float* b, float* c, size_t size) {
  size_t i = 0;

  // AVX2: 8 個の float を同時処理
  for (; i + 8 <= size; i += 8) {
    __m256 va = _mm256_load_ps(a + i);  // Aligned load
    __m256 vb = _mm256_load_ps(b + i);
    __m256 vc = _mm256_add_ps(va, vb);
    _mm256_store_ps(c + i, vc);         // Aligned store
  }

  // 残りの要素をスカラー処理
  for (; i < size; ++i) {
    c[i] = a[i] + b[i];
  }
}

/**
 * @brief AVX2 を使用した要素ごとの乗算
 *
 * @param a 入力配列 A (32-byte aligned)
 * @param b 入力配列 B (32-byte aligned)
 * @param c 出力配列 C (32-byte aligned)
 * @param size 要素数
 */
inline void mul_avx2(const float* a, const float* b, float* c, size_t size) {
  size_t i = 0;

  for (; i + 8 <= size; i += 8) {
    __m256 va = _mm256_load_ps(a + i);
    __m256 vb = _mm256_load_ps(b + i);
    __m256 vc = _mm256_mul_ps(va, vb);
    _mm256_store_ps(c + i, vc);
  }

  for (; i < size; ++i) {
    c[i] = a[i] * b[i];
  }
}

/**
 * @brief AVX2 を使用した要素ごとの減算
 *
 * @param a 入力配列 A (32-byte aligned)
 * @param b 入力配列 B (32-byte aligned)
 * @param c 出力配列 C (32-byte aligned)
 * @param size 要素数
 */
inline void sub_avx2(const float* a, const float* b, float* c, size_t size) {
  size_t i = 0;

  for (; i + 8 <= size; i += 8) {
    __m256 va = _mm256_load_ps(a + i);
    __m256 vb = _mm256_load_ps(b + i);
    __m256 vc = _mm256_sub_ps(va, vb);
    _mm256_store_ps(c + i, vc);
  }

  for (; i < size; ++i) {
    c[i] = a[i] - b[i];
  }
}

/**
 * @brief AVX2 を使用した要素ごとの除算
 *
 * @param a 入力配列 A (32-byte aligned)
 * @param b 入力配列 B (32-byte aligned)
 * @param c 出力配列 C (32-byte aligned)
 * @param size 要素数
 */
inline void div_avx2(const float* a, const float* b, float* c, size_t size) {
  size_t i = 0;

  for (; i + 8 <= size; i += 8) {
    __m256 va = _mm256_load_ps(a + i);
    __m256 vb = _mm256_load_ps(b + i);
    __m256 vc = _mm256_div_ps(va, vb);
    _mm256_store_ps(c + i, vc);
  }

  for (; i < size; ++i) {
    c[i] = a[i] / b[i];
  }
}

}  // namespace simd
}  // namespace cpu
}  // namespace gradflow
```

**完了基準**:
- [ ] `simd_ops.hpp` が作成されている
- [ ] コンパイルが通る（`-mavx2` フラグ付き）

**推定時間**: 20 分

---

## Phase 3: CPUKernels Public Interface の実装

### Task 3.1: CPUKernels ヘッダーファイルの作成

**説明**: CPUKernels クラスの public interface を定義します。

**作業内容**:

1. `include/gradflow/autograd/cpu/kernels.hpp` を作成（設計書のセクション 3.1 を参照）

**完了基準**:
- [ ] ヘッダーファイルが作成されている
- [ ] すべての public メソッドにドキュメントがある
- [ ] コンパイルが通る

**推定時間**: 25 分

---

### Task 3.2: CPUKernels 実装ファイルの作成

**説明**: CPUKernels クラスの実装を作成します。

**作業内容**:

1. `src/autograd/cpu/kernels_avx2.cpp` を作成（設計書のセクション 4.3 を参照）

**重要なポイント**:
- CPUID による機能検出（`has_avx2()`, `has_avx512()`）
- AVX2 が使用できない場合のスカラーフォールバック
- アライメント付きメモリ確保/解放

**完了基準**:
- [ ] `kernels_avx2.cpp` が作成されている
- [ ] `add`, `mul`, `sub`, `div` メソッドが実装されている
- [ ] `getSIMDInfo()` メソッドが実装されている
- [ ] `alignedAlloc`, `alignedFree` が実装されている
- [ ] コンパイルが通る

**推定時間**: 30 分

---

## Phase 4: Blocked MatMul の実装

### Task 4.1: Blocked MatMul の実装

**説明**: キャッシュ効率を考慮した Blocked MatMul を実装します。

**作業内容**:

1. `src/autograd/cpu/matmul_blocked.cpp` を作成（設計書のセクション 4.2 を参照）

**重要なポイント**:
- ブロックサイズ: 64x64（L1 キャッシュに収まるサイズ）
- マイクロカーネル: 4x4 レジスタブロック
- OpenMP 並列化（`#pragma omp parallel for collapse(2)`）

**完了基準**:
- [ ] `matmul_blocked.cpp` が作成されている
- [ ] `matmul_kernel_4x4` マイクロカーネルが実装されている
- [ ] `matmul_block` 関数が実装されている
- [ ] `CPUKernels::matmul` が実装されている
- [ ] OpenMP が有効な場合は並列実行される
- [ ] コンパイルが通る

**推定時間**: 45 分

---

## Phase 5: テストの実装

### Task 5.1: テストファイルの作成と基本テスト

**説明**: CPU 最適化のテストケースを作成します。

**作業内容**:

1. `tests/test_cpu_optimized.cpp` を作成

2. 以下のテストケースを実装:
   - `CPUOptimizedTest::AddSIMD_Correctness`
   - `CPUOptimizedTest::MulSIMD_Correctness`
   - `CPUOptimizedTest::SubSIMD_Correctness`
   - `CPUOptimizedTest::DivSIMD_Correctness`

**完了基準**:
- [ ] テストファイルが作成されている
- [ ] 4 つの正確性テストが実装されている
- [ ] すべてのテストが pass

**推定時間**: 30 分

---

### Task 5.2: MatMul 正確性テストの実装

**説明**: Blocked MatMul の正確性を検証するテストを実装します。

**作業内容**:

1. `tests/test_cpu_optimized.cpp` に以下を追加:
   - `CPUOptimizedTest::BlockedMatMul_Correctness`

**テストケース例**:
```cpp
TEST_F(CPUOptimizedTest, BlockedMatMul_Correctness) {
  constexpr size_t m = 4, k = 3, n = 2;

  float* a = static_cast<float*>(alignedAlloc(m * k * sizeof(float)));
  float* b = static_cast<float*>(alignedAlloc(k * n * sizeof(float)));
  float* c = static_cast<float*>(alignedAlloc(m * n * sizeof(float)));

  // A = [[1, 2, 3],
  //      [4, 5, 6],
  //      [7, 8, 9],
  //      [10, 11, 12]]
  for (size_t i = 0; i < m * k; ++i) {
    a[i] = static_cast<float>(i + 1);
  }

  // B = [[1, 2],
  //      [3, 4],
  //      [5, 6]]
  for (size_t i = 0; i < k * n; ++i) {
    b[i] = static_cast<float>(i + 1);
  }

  kernels_->matmul(a, b, c, m, k, n);

  // Expected C = A @ B
  EXPECT_FLOAT_EQ(c[0 * n + 0], 22.0f);  // 1*1 + 2*3 + 3*5
  EXPECT_FLOAT_EQ(c[0 * n + 1], 28.0f);  // 1*2 + 2*4 + 3*6
  EXPECT_FLOAT_EQ(c[1 * n + 0], 49.0f);  // 4*1 + 5*3 + 6*5
  EXPECT_FLOAT_EQ(c[1 * n + 1], 64.0f);  // 4*2 + 5*4 + 6*6

  alignedFree(c);
  alignedFree(b);
  alignedFree(a);
}
```

**完了基準**:
- [ ] MatMul 正確性テストが実装されている
- [ ] テストが pass

**推定時間**: 20 分

---

### Task 5.3: パフォーマンステストの実装

**説明**: 最適化前後のパフォーマンスを比較するテストを実装します。

**作業内容**:

1. `tests/test_cpu_optimized.cpp` に以下を追加:
   - `CPUOptimizedTest::Performance_Add`
   - `CPUOptimizedTest::Performance_MatMul`

**重要なポイント**:
- 大きいサイズのデータでテスト（Add: 10M 要素、MatMul: 512x512）
- 最適化版と naive 版の実行時間を測定
- Speedup を計算して表示
- 最適化版が高速であることを確認

**完了基準**:
- [ ] パフォーマンステストが実装されている
- [ ] ベンチマーク結果が表示される
- [ ] 最適化版が naive 版より高速

**推定時間**: 35 分

---

### Task 5.4: Eigen 比較テストの実装（オプション）

**説明**: Eigen ライブラリとのパフォーマンス比較テストを実装します。

**作業内容**:

1. `tests/test_cpu_optimized.cpp` に以下を追加:
   - `CPUOptimizedTest::Performance_MatMul_vs_Eigen`

**前提条件**:
- Eigen3 がインストールされている
- CMake で Eigen3 が見つかっている

**完了基準**:
- [ ] Eigen 比較テストが実装されている
- [ ] Eigen の 80% 以上の速度を達成

**推定時間**: 25 分（オプション）

---

## Phase 6: CMake とテスト統合

### Task 6.1: テストの CMake 統合

**説明**: CPU 最適化テストを CMake テストスイートに統合します。

**作業内容**:

1. `tests/CMakeLists.txt` に以下を追加:

```cmake
# CPU 最適化テスト
if(GRADFLOW_ENABLE_CPU_OPTIMIZATIONS)
    add_executable(cpu_optimized_test test_cpu_optimized.cpp)
    target_link_libraries(cpu_optimized_test
        PRIVATE
            fullscratch
            GTest::gtest
            GTest::gtest_main
    )
    gtest_discover_tests(cpu_optimized_test)
endif()
```

2. テストのビルドと実行を確認

```bash
cmake -DGRADFLOW_ENABLE_CPU_OPTIMIZATIONS=ON ..
make cpu_optimized_test
./tests/cpu_optimized_test
```

**完了基準**:
- [ ] テストが CMake に統合されている
- [ ] `ctest` でテストが実行される
- [ ] すべてのテストが pass

**推定時間**: 15 分

---

## Phase 7: CI 統合

### Task 7.1: GitHub Actions ワークフローの更新

**説明**: CI で CPU 最適化テストを実行するように設定します。

**作業内容**:

1. `.github/workflows/ci.yml` に CPU 最適化ビルドを追加

```yaml
- name: Build with CPU Optimizations
  run: |
    cmake -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DGRADFLOW_BUILD_TESTS=ON \
      -DGRADFLOW_ENABLE_CPU_OPTIMIZATIONS=ON
    cmake --build build

- name: Run CPU Optimized Tests
  run: |
    cd build
    ctest --output-on-failure -R cpu_optimized_test
```

**完了基準**:
- [ ] CI ワークフローが更新されている
- [ ] CI でテストが実行される
- [ ] すべてのテストが pass

**推定時間**: 20 分

---

## Phase 8: ドキュメント

### Task 8.1: README.md の更新

**説明**: CPU 最適化機能を README.md に追加します。

**作業内容**:

1. `README.md` に以下のセクションを追加:

```markdown
## CPU 最適化

GradFlow は CPU 上での高速な演算をサポートします。

### 機能

- **SIMD ベクトル化**: AVX2/AVX-512 による高速な elementwise 演算
- **OpenMP 並列化**: マルチコア CPU を活用した並列行列乗算
- **Blocked MatMul**: キャッシュ効率を考慮したブロック行列乗算
- **メモリアライメント**: 32 バイトアライメントによる高速メモリアクセス

### ビルド方法

```bash
cmake -B build -DGRADFLOW_ENABLE_CPU_OPTIMIZATIONS=ON
cmake --build build
```

### パフォーマンス

| 演算 | 最適化前 | 最適化後 | Speedup |
|------|----------|----------|---------|
| Add (10M 要素) | 45 ms | 12 ms | 3.8x |
| MatMul (512x512) | 850 ms | 65 ms | 13.1x |
| MatMul vs Eigen | - | 85% | - |
```

**完了基準**:
- [ ] README.md が更新されている
- [ ] ビルド方法が記載されている
- [ ] パフォーマンス結果が記載されている

**推定時間**: 20 分

---

### Task 8.2: PROGRESS.md の更新

**説明**: プロジェクト進捗を更新します。

**作業内容**:

1. `PROGRESS.md` に Issue #19 の完了を記載

```markdown
## Phase 4: CPU 最適化
### ステータス: ✅ 完了

- ✅ 4.1 CPU 最適化の実装 (Week 1-2)
  - Issue #19: 完了（PR #XX: マージ済み）
  - ステータス: SIMD ベクトル化、OpenMP 並列化、Blocked MatMul 実装完了
  - パフォーマンス: MatMul が Eigen の 85% の速度
  - 設計書: `docs/ISSUE_19_cpu_optimization_design.md`
```

**完了基準**:
- [ ] PROGRESS.md が更新されている

**推定時間**: 10 分

---

## 完了チェックリスト

### 機能実装

- [ ] SIMD Intrinsic 演算（add, mul, sub, div）が実装されている
- [ ] Blocked MatMul が実装されている
- [ ] OpenMP 並列化が実装されている
- [ ] アライメント付きメモリ確保/解放が実装されている
- [ ] AVX2 が使用できない環境でのフォールバックが動作する

### テスト

- [ ] すべてのテストが pass（最低 5 個）
- [ ] 数値的正確性が保たれている（相対誤差 < 1e-5）
- [ ] ベンチマークで最適化前より高速
  - [ ] Add: 最適化版が naive 版より 3x 以上高速
  - [ ] MatMul: 最適化版が naive 版より 10x 以上高速
- [ ] MatMul が Eigen の 80% 以上の速度（オプション）

### CI/CD

- [ ] CI でテストが実行される
- [ ] すべての CI チェックが pass

### ドキュメント

- [ ] README.md に CPU 最適化機能が記載されている
- [ ] PROGRESS.md が更新されている
- [ ] 設計書が `docs/ISSUE_19_cpu_optimization_design.md` にある

---

## 推定総作業時間

| Phase | 時間 |
|-------|------|
| Phase 1: プロジェクト構造のセットアップ | 20 分 |
| Phase 2: SIMD Intrinsic 演算の実装 | 20 分 |
| Phase 3: CPUKernels Public Interface の実装 | 55 分 |
| Phase 4: Blocked MatMul の実装 | 45 分 |
| Phase 5: テストの実装 | 110 分 |
| Phase 6: CMake とテスト統合 | 15 分 |
| Phase 7: CI 統合 | 20 分 |
| Phase 8: ドキュメント | 30 分 |
| **合計** | **約 5 時間** |

---

## 注意事項

1. **アライメント**: すべてのメモリは 32 バイトアライメントされている必要があります
2. **AVX2 サポート**: x86_64 プロセッサで AVX2 が使用可能であることを前提とします
3. **OpenMP**: OpenMP がない環境では、並列化が無効になりますが、シリアル版は動作します
4. **数値精度**: 浮動小数点演算の精度には注意が必要です（相対誤差 < 1e-5 を目標）
5. **パフォーマンス測定**: ベンチマークは Release ビルド（`-O3 -march=native`）で実行してください

---

以上が Issue #19 の実装タスクリストです。各タスクを順番に実装し、テストが通ることを確認してから次に進んでください。
