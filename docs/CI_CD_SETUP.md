# CI/CD セットアップガイド

## 目次

- [概要](#概要)
- [パイプライン構成](#パイプライン構成)
- [ワークフロー詳細](#ワークフロー詳細)
- [ローカルでの実行](#ローカルでの実行)
- [トラブルシューティング](#トラブルシューティング)
- [ベストプラクティス](#ベストプラクティス)

---

## 概要

GradFlow の CI/CD パイプラインは、コードの品質、テストカバレッジ、パフォーマンスを自動的に検証し、プロダクションレディなコードを保証します。

### 設計原則

1. **自動化**: すべてのチェックは自動的に実行される
2. **再現性**: ML ワークフローの再現性を確保するためのシード固定
3. **並列化**: 複数の環境で並列にテストを実行
4. **段階的検証**: リント → 型チェック → テスト → ベンチマーク
5. **早期失敗**: 問題を早期に検出して修正コストを削減

### サポートプラットフォーム

- **OS**: Ubuntu, macOS, Windows
- **Python**: 3.9, 3.10, 3.11, 3.12
- **C++ コンパイラ**: GCC 11+, Clang 15+, MSVC 193+
- **GPU**: Metal (macOS), CUDA (オプション)

---

## パイプライン構成

### 1. メイン CI パイプライン (.github/workflows/ci.yml)

**トリガー条件**:
- `main` または `develop` ブランチへの push
- `main` または `develop` ブランチへの pull request
- 毎日 0:00 UTC（定期的な健全性チェック）

**ジョブ**:

#### 1.1 build-and-test
複数の OS、コンパイラ、ビルドタイプでビルドとテストを実行

**マトリックス**:
```yaml
os: [ubuntu-latest, macos-latest, windows-latest]
compiler: [gcc, clang, msvc]
build_type: [Debug, Release]
```

**ステップ**:
1. コードのチェックアウト（サブモジュールを含む）
2. Python 3.11 のセットアップ
3. Conan 2.0.17 のインストール
4. 依存関係のインストール
5. CMake 設定（Ninja ビルドシステム）
6. ビルド（並列実行）
7. テスト実行（CTest）
   - タイムアウト: 300秒
   - ランダムシャッフル実行
   - 失敗時は最大 2 回リトライ
8. カバレッジレポート生成（Debug + GCC のみ）
9. Codecov へアップロード

**ML 固有の設定**:
```bash
# 再現性のための環境変数
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

#### 1.2 sanitizer-tests
メモリリーク、未定義動作、データ競合を検出

**サニタイザー**:
- **AddressSanitizer**: メモリリークとバッファオーバーフロー
- **UndefinedBehaviorSanitizer**: 未定義動作
- **ThreadSanitizer**: データ競合

**環境変数**:
```bash
ASAN_OPTIONS=detect_leaks=1:symbolize=1
UBSAN_OPTIONS=print_stacktrace=1
TSAN_OPTIONS=second_deadlock_stack=1
```

---

### 2. Python バインディングテスト (.github/workflows/python_tests.yml)

**トリガー条件**:
- `python/**`, `include/**`, `src/**` の変更時

**ジョブ**:

#### 2.1 python-bindings
Python バインディングのテストとリント

**マトリックス**:
```yaml
os: [ubuntu-latest, macos-latest, windows-latest]
python-version: ['3.9', '3.10', '3.11', '3.12']
```

**ステップ**:
1. Python バインディングのビルド（nanobind 使用）
2. `pip install -e .` でインストール
3. pytest によるテスト実行
   - カバレッジレポート生成
   - 並列実行（`-n auto`）
4. mypy による型チェック（strict モード）
5. flake8 によるリント
6. black によるフォーマットチェック
7. isort による import 整理チェック

#### 2.2 build-wheels
cibuildwheel を使用して Python wheel をビルド

**対象プラットフォーム**:
- Linux: x86_64
- macOS: x86_64, arm64（Apple Silicon）
- Windows: AMD64

**ビルドされる wheel**:
- Python 3.9 - 3.12
- manylinux, macosx, win_amd64

#### 2.3 test-import
ビルドされた wheel のインポートテスト

#### 2.4 publish-test-pypi
Test PyPI への自動パブリッシュ（main ブランチのみ）

#### 2.5 python-documentation
Sphinx によるドキュメント生成と GitHub Pages へのデプロイ

---

### 3. コード品質チェック (.github/workflows/code_quality.yml)

**ジョブ**:

#### 3.1 clang-format
C++ コードのフォーマットチェック（`.clang-format` に基づく）

#### 3.2 clang-tidy
C++ の静的解析（`.clang-tidy` に基づく）

**チェック項目**:
- バグ検出（bugprone-*）
- セキュリティ（cert-*）
- C++ Core Guidelines（cppcoreguidelines-*）
- Google スタイル（google-*）
- パフォーマンス（performance-*）
- 可読性（readability-*）

#### 3.3 cppcheck
追加の C++ 静的解析

#### 3.4 include-what-you-use
ヘッダーインクルードの最適化チェック

#### 3.5 cmake-format
CMakeLists.txt のフォーマットチェック

#### 3.6 complexity-analysis
関数の循環的複雑度チェック（lizard）

**閾値**:
- 複雑度: 15 以下
- 行数: 1000 以下

#### 3.7 security-scan
セキュリティ脆弱性スキャン（Trivy）

---

### 4. Metal GPU テスト (.github/workflows/metal_tests.yml)

**トリガー条件**:
- Metal 関連コードの変更時

**実行環境**: macOS（Apple Silicon 推奨）

**テスト項目**:
- Metal デバイスの初期化
- Metal カーネルの実行
- CPU vs Metal の数値一致性
- メモリ管理（Unified Memory）
- MPS（Metal Performance Shaders）の動作確認

---

### 5. CUDA GPU テスト (.github/workflows/cuda_tests.yml)

**トリガー条件**:
- CUDA 関連コードの変更時

**実行環境**: Self-hosted runner（NVIDIA GPU 搭載）

**テスト項目**:
- CUDA デバイスの初期化
- CUDA カーネルの実行
- CPU vs CUDA の数値一致性
- cuBLAS による行列演算
- メモリ管理

---

### 6. ベンチマーク (.github/workflows/benchmarks.yml)

**トリガー条件**:
- 手動実行（workflow_dispatch）
- 週次（日曜日 0:00 UTC）

**ベンチマーク項目**:
- MatMul（CPU, Metal, CUDA）
- Transformer forward/backward
- メモリ使用量
- 勾配計算速度

**結果の保存**:
- GitHub Actions アーティファクト
- パフォーマンス回帰の自動検出

---

## ワークフロー詳細

### 依存関係のインストール

#### C++ 依存関係（Conan 2.0）

```bash
# Conan プロファイルの検出
conan profile detect --force

# 依存関係のインストール
conan install . \
  --output-folder=build \
  --build=missing \
  --settings=build_type=Release
```

**主要な依存関係**:
- Eigen3: 線形代数
- Google Test: テストフレームワーク
- Google Benchmark: ベンチマーク

#### Python 依存関係

```bash
# 開発用依存関係のインストール
pip install -r requirements-dev.txt
```

**主要な依存関係**:
- NumPy: 数値計算
- pytest: テストフレームワーク
- mypy: 型チェック
- black: フォーマッター
- flake8: リンター

---

### ビルドプロセス

#### CMake 設定

```bash
cmake -B build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake \
  -DGRADFLOW_BUILD_TESTS=ON \
  -DGRADFLOW_BUILD_PYTHON_BINDINGS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

**重要なオプション**:
- `CMAKE_EXPORT_COMPILE_COMMANDS`: clang-tidy 用
- `GRADFLOW_ENABLE_COVERAGE`: カバレッジ測定
- `GRADFLOW_ENABLE_SANITIZER`: サニタイザー有効化

#### ビルド実行

```bash
cmake --build build --config Release --parallel
```

**環境変数**:
```bash
export CMAKE_BUILD_PARALLEL_LEVEL=4
```

---

### テスト実行

#### C++ テスト（CTest）

```bash
cd build
ctest --build-config Release \
  --output-on-failure \
  --parallel \
  --timeout 300 \
  --schedule-random \
  --repeat until-pass:2
```

**オプション**:
- `--output-on-failure`: 失敗時のみ出力
- `--parallel`: 並列実行
- `--timeout 300`: 300秒でタイムアウト
- `--schedule-random`: ランダム順で実行
- `--repeat until-pass:2`: 失敗時は最大2回リトライ

#### Python テスト（pytest）

```bash
pytest python/tests \
  -v \
  --cov=gradflow \
  --cov-report=term-missing \
  --cov-report=html \
  --cov-report=xml \
  --cov-fail-under=80 \
  -n auto \
  --timeout=300
```

**オプション**:
- `-v`: 詳細出力
- `--cov`: カバレッジ測定
- `--cov-fail-under=80`: 80% 未満で失敗
- `-n auto`: 並列実行（CPU コア数に応じて自動）
- `--timeout=300`: 300秒でタイムアウト

---

### コードカバレッジ

#### C++ カバレッジ（lcov）

```bash
# カバレッジ情報の収集
lcov --directory build --capture --output-file coverage.info

# 不要なファイルを除外
lcov --remove coverage.info '/usr/*' '*/tests/*' '*/external/*' --output-file coverage.info

# レポート表示
lcov --list coverage.info

# Codecov へアップロード
codecov --file coverage.info
```

#### Python カバレッジ（coverage.py）

```bash
# pytest 実行時に自動的に測定される
pytest --cov=gradflow --cov-report=xml

# Codecov へアップロード
codecov --file coverage.xml
```

---

### 静的解析

#### clang-tidy

```bash
# compile_commands.json を使用
clang-tidy-15 \
  -p build/compile_commands.json \
  --config-file=.clang-tidy \
  --warnings-as-errors='*' \
  src/**/*.cpp include/**/*.hpp
```

#### cppcheck

```bash
cppcheck \
  --enable=all \
  --error-exitcode=1 \
  --std=c++17 \
  --xml \
  --xml-version=2 \
  --output-file=cppcheck-report.xml \
  -I include \
  src
```

#### mypy（Python）

```bash
mypy python/gradflow --strict
```

---

## ローカルでの実行

### Pre-commit フックのセットアップ

```bash
# pre-commit のインストール
pip install pre-commit

# フックのインストール
pre-commit install

# 全ファイルに対して実行
pre-commit run --all-files
```

**実行されるチェック**:
1. C++ フォーマット（clang-format）
2. CMake フォーマット（cmake-format）
3. Python フォーマット（black, isort）
4. Python リント（flake8）
5. Python 型チェック（mypy）
6. 一般的なファイルチェック（trailing whitespace, EOF, など）
7. シークレット検出（detect-secrets）
8. Markdown フォーマット（mdformat）

### ローカルでのテスト実行

#### C++ テスト

```bash
# ビルド
mkdir build && cd build
cmake .. -DGRADFLOW_BUILD_TESTS=ON
cmake --build . --parallel

# テスト実行
ctest --output-on-failure
```

#### Python テスト

```bash
# 依存関係のインストール
pip install -r requirements-dev.txt

# テスト実行
pytest python/tests -v
```

### ローカルでのリント実行

#### C++

```bash
# clang-format（自動修正）
find include src -name '*.cpp' -o -name '*.hpp' | \
  xargs clang-format -i

# clang-tidy（検査のみ）
clang-tidy -p build/compile_commands.json src/**/*.cpp
```

#### Python

```bash
# black（自動修正）
black python/gradflow python/tests

# isort（自動修正）
isort python/gradflow python/tests

# flake8（検査のみ）
flake8 python/gradflow python/tests

# mypy（検査のみ）
mypy python/gradflow --strict
```

---

## トラブルシューティング

### よくある問題

#### 1. Conan の依存関係エラー

**症状**: `ERROR: Missing prebuilt package for ...`

**解決策**:
```bash
# 依存関係を強制的にビルド
conan install . --build=missing
```

#### 2. CMake の設定エラー

**症状**: `CMake Error: Could not find ...`

**解決策**:
```bash
# ビルドディレクトリをクリーン
rm -rf build
mkdir build && cd build

# Conan から再インストール
conan install .. --build=missing
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
```

#### 3. Python バインディングのビルドエラー

**症状**: `ModuleNotFoundError: No module named 'nanobind'`

**解決策**:
```bash
# nanobind を FetchContent で自動取得するように設定されているが、
# 手動でインストールする場合：
pip install nanobind
```

#### 4. テストのタイムアウト

**症状**: `Test #X: Timeout`

**解決策**:
```bash
# タイムアウトを延長
ctest --timeout 600

# または、特定のテストのみ実行
ctest -R test_name
```

#### 5. カバレッジ閾値未達

**症状**: `Coverage threshold (80%) not met`

**解決策**:
1. カバレッジレポートを確認: `htmlcov/index.html`
2. テストされていないコードを特定
3. 単体テストを追加

#### 6. ML テストの非決定性

**症状**: テスト結果が実行ごとに異なる

**解決策**:
```python
# テスト内でランダムシードを固定
import numpy as np
import random

def setup_module():
    random.seed(42)
    np.random.seed(42)
```

---

## ベストプラクティス

### 1. コミット前のチェック

```bash
# pre-commit フックを必ず実行
pre-commit run --all-files

# ローカルでテストを実行
pytest python/tests -v
ctest --test-dir build
```

### 2. Pull Request のタイトル

Conventional Commits 形式に従う:

```
feat: Add Metal GPU support for MatMul
fix: Fix memory leak in Tensor deallocation
docs: Update CI/CD setup guide
test: Add property-based tests for gradient computation
refactor: Simplify Operation base class
perf: Optimize MatMul with blocked algorithm
```

### 3. テストの分類

```python
import pytest

# 単体テスト
@pytest.mark.unit
def test_tensor_creation():
    ...

# 統合テスト
@pytest.mark.integration
def test_end_to_end_training():
    ...

# GPU テスト
@pytest.mark.metal
def test_metal_matmul():
    ...

# 遅いテスト
@pytest.mark.slow
def test_large_model_training():
    ...

# 数値勾配チェック
@pytest.mark.numerical
def test_gradient_accuracy():
    ...
```

### 4. 再現性の確保

#### ランダムシードの固定

```python
# Python
import random
import numpy as np

random.seed(42)
np.random.seed(42)
```

```cpp
// C++
#include <random>

std::mt19937 gen(42);  // 固定シード
```

#### 環境変数の設定

```bash
# スレッド数を固定
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ハッシュシードを固定
export PYTHONHASHSEED=0
```

### 5. パフォーマンステスト

```python
import pytest

@pytest.mark.benchmark
def test_matmul_performance(benchmark):
    def matmul():
        # 行列積の実行
        ...

    result = benchmark(matmul)

    # パフォーマンス閾値を設定
    assert result.stats.mean < 1.0  # 1秒以内
```

### 6. プロパティベーステスト

```python
from hypothesis import given, strategies as st
import pytest

@pytest.mark.property
@given(
    shape=st.tuples(
        st.integers(min_value=1, max_value=100),
        st.integers(min_value=1, max_value=100)
    )
)
def test_tensor_properties(shape):
    """Tensor の不変条件をテスト"""
    tensor = Tensor.zeros(shape)

    # プロパティ1: shape が正しい
    assert tensor.shape == shape

    # プロパティ2: size が一致
    assert tensor.size() == shape[0] * shape[1]
```

### 7. 数値勾配チェック

```python
@pytest.mark.numerical
def test_gradient_numerical_check():
    """自動微分と数値微分の一致を検証"""
    def numerical_gradient(f, x, eps=1e-5):
        grad = np.zeros_like(x)
        for i in range(x.size):
            x_plus = x.copy()
            x_plus.flat[i] += eps
            x_minus = x.copy()
            x_minus.flat[i] -= eps
            grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
        return grad

    # 自動微分
    x = Variable(Tensor([2.0, 3.0]), requires_grad=True)
    y = (x ** 2).sum()
    y.backward()
    auto_grad = x.grad.data

    # 数値微分
    num_grad = numerical_gradient(lambda x: (x ** 2).sum(), x.data)

    # 比較（相対誤差 < 1e-5）
    np.testing.assert_allclose(auto_grad, num_grad, rtol=1e-5, atol=1e-7)
```

---

## CI/CD パイプラインの監視

### GitHub Actions の確認

1. リポジトリの **Actions** タブを開く
2. 各ワークフローの実行結果を確認
3. 失敗したジョブのログを確認

### Codecov の確認

1. [Codecov ダッシュボード](https://codecov.io/) にアクセス
2. カバレッジの推移を確認
3. カバーされていないコードを特定

### パフォーマンス回帰の監視

1. Benchmark ワークフローの結果を確認
2. 前回実行との比較
3. 回帰が検出された場合はアラート

---

## 参考資料

### GitHub Actions

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)

### テストフレームワーク

- [Google Test Documentation](https://google.github.io/googletest/)
- [pytest Documentation](https://docs.pytest.org/)
- [hypothesis Documentation](https://hypothesis.readthedocs.io/)

### 静的解析

- [clang-tidy Documentation](https://clang.llvm.org/extra/clang-tidy/)
- [cppcheck Manual](http://cppcheck.sourceforge.net/manual.pdf)
- [mypy Documentation](https://mypy.readthedocs.io/)

### コードカバレッジ

- [lcov Documentation](http://ltp.sourceforge.net/coverage/lcov.php)
- [coverage.py Documentation](https://coverage.readthedocs.io/)
- [Codecov Documentation](https://docs.codecov.com/)

---

## まとめ

GradFlow の CI/CD パイプラインは、以下を保証します:

1. **コード品質**: リント、型チェック、静的解析
2. **テストカバレッジ**: 80% 以上のカバレッジ
3. **クロスプラットフォーム**: Ubuntu, macOS, Windows
4. **再現性**: ML ワークフローの決定論的動作
5. **パフォーマンス**: ベンチマークによる回帰検出
6. **セキュリティ**: 脆弱性スキャン

すべての変更は、これらのチェックをパスしてからマージされます。
