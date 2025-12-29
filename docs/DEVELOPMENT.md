# GradFlow 開発ガイド

## 目次

- [開発環境のセットアップ](#開発環境のセットアップ)
- [ツールチェーン](#ツールチェーン)
- [タスクランナー（cargo-make）](#タスクランナーcargo-make)
- [コード品質チェック](#コード品質チェック)
- [テスト](#テスト)
- [ビルド](#ビルド)
- [ドキュメント](#ドキュメント)
- [ワークフロー](#ワークフロー)

---

## 開発環境のセットアップ

### 前提条件

- **Python**: 3.9 以上
- **C++ コンパイラ**: GCC 11+, Clang 15+, または MSVC 193+
- **CMake**: 3.20 以上
- **Rust**: 最新安定版（cargo-make のため）

### クイックスタート

```bash
# 1. リポジトリのクローン
git clone https://github.com/yourusername/gradflow.git
cd gradflow

# 2. cargo-make のインストール
cargo install --locked cargo-make

# 3. 開発環境のセットアップ
makers dev
```

`makers dev` コマンドは以下を実行します：
- uv のインストール
- Python 依存関係のインストール
- pre-commit フックの設定

---

## ツールチェーン

GradFlow は現代的な Python ツールチェーンを採用しています。

### uv - 高速パッケージマネージャー

[uv](https://github.com/astral-sh/uv) は Rust で書かれた高速な Python パッケージマネージャーです。

**インストール**:
```bash
# 自動インストール（makers dev が実行）
curl -LsSf https://astral.sh/uv/install.sh | sh

# または手動インストール
makers install-uv
```

**使用方法**:
```bash
# 依存関係のインストール
uv pip install -e ".[dev]"

# パッケージの追加
uv pip install <package-name>

# 依存関係の更新
uv pip install --upgrade -e ".[dev]"
```

### Ruff - 高速リンター/フォーマッター

[Ruff](https://github.com/astral-sh/ruff) は Rust で書かれた高速な Python リンター/フォーマッターです。black, isort, flake8 などを置き換えます。

**機能**:
- コードフォーマット（black 互換）
- import 整理（isort 互換）
- リント（flake8, pylint などのルールセット）
- セキュリティチェック（bandit 互換）

**設定**: `pyproject.toml` の `[tool.ruff]` セクション

**使用方法**:
```bash
# フォーマット
makers format

# フォーマットチェック（変更なし）
makers format-check

# リント
makers lint

# リント + 自動修正
makers lint-fix
```

### Pyright - 高速型チェッカー

[Pyright](https://github.com/microsoft/pyright) は Microsoft が開発した高速な静的型チェッカーです。mypy よりも高速で、strict モードをサポートします。

**特徴**:
- mypy より高速
- VSCode 統合（Pylance）
- 厳格な型チェック

**設定**: `pyproject.toml` の `[tool.pyright]` セクション

**使用方法**:
```bash
# 型チェック
makers typecheck
```

---

## タスクランナー（cargo-make）

GradFlow は [cargo-make](https://github.com/sagiegurari/cargo-make) をタスクランナーとして使用しています。すべての開発タスクを `makers` コマンドから実行できます。

### 基本コマンド

```bash
# ヘルプを表示
makers help
makers --list-all-steps

# 開発環境のセットアップ
makers dev

# 依存関係のインストール
makers install
```

### Python コード品質

```bash
# コードフォーマット
makers format

# フォーマットチェック（CI で使用）
makers format-check

# リント
makers lint

# リント + 自動修正
makers lint-fix

# 型チェック
makers typecheck

# 全チェック（format-check + lint + typecheck）
makers check

# コード品質チェック（format + lint-fix + typecheck）
makers qa
```

### テスト

```bash
# Python テスト
makers test

# 高速テスト（並列実行なし）
makers test-fast

# 単体テストのみ
makers test-unit

# 統合テストのみ
makers test-integration

# GPU テスト
makers test-gpu

# C++ テスト
makers test-cpp

# 全テスト（Python + C++）
makers test-all

# カバレッジレポート
makers coverage
```

### C++ ビルド

```bash
# Release ビルド
makers build

# Debug ビルド
makers build-debug

# Release ビルド（最適化オプション付き）
makers build-release

# ビルドディレクトリをクリーン
makers clean-build
```

### C++ コード品質

```bash
# C++ フォーマット
makers format-cpp

# C++ フォーマットチェック
makers format-cpp-check

# C++ リント（clang-tidy）
makers lint-cpp
```

### ドキュメント

```bash
# ドキュメント生成
makers docs

# ドキュメントサーバー起動（localhost:8000）
makers docs-serve
```

### パフォーマンス

```bash
# ベンチマーク実行
makers bench

# プロファイリング
makers profile
```

### クリーンアップ

```bash
# 生成物をクリーン
makers clean

# 全てをクリーン（依存関係含む）
makers clean-all
```

### CI/CD

```bash
# CI で実行する全チェック
makers ci
```

---

## コード品質チェック

### Python

GradFlow は厳格なコード品質基準を採用しています。

#### フォーマット（Ruff）

- **ライン長**: 88 文字
- **引用符**: ダブルクォート
- **インデント**: スペース 4 つ

```bash
# フォーマット実行
makers format

# フォーマットチェック
makers format-check
```

#### リント（Ruff）

有効なルールセット:
- `E`, `W`: pycodestyle
- `F`: pyflakes
- `I`: isort（import 整理）
- `B`: flake8-bugbear
- `C4`: flake8-comprehensions
- `UP`: pyupgrade
- `SIM`: flake8-simplify
- `N`: pep8-naming
- `D`: pydocstyle（Google スタイル）
- `PL`: pylint
- `RUF`: ruff 固有のルール
- `ANN`: flake8-annotations
- `S`: flake8-bandit（セキュリティ）
- `A`: flake8-builtins
- `COM`: flake8-commas
- `C90`: mccabe complexity
- `T20`: flake8-print

```bash
# リント実行
makers lint

# リント + 自動修正
makers lint-fix
```

#### 型チェック（Pyright）

- **モード**: strict
- **Python バージョン**: 3.9
- すべての型エラーを検出

```bash
# 型チェック実行
makers typecheck
```

### C++

#### フォーマット（clang-format）

- **スタイル**: Google スタイルベース
- **設定**: `.clang-format`

```bash
# C++ フォーマット
makers format-cpp

# フォーマットチェック
makers format-cpp-check
```

#### リント（clang-tidy）

- **設定**: `.clang-tidy`
- **ルール**: modernize, performance, readability など

```bash
# C++ リント
makers lint-cpp
```

---

## テスト

### Python テスト（pytest）

#### テスト構造

```
python/tests/
├── unit/           # 単体テスト
├── integration/    # 統合テスト
├── gpu/            # GPU テスト
└── benchmarks/     # ベンチマーク
```

#### テストマーカー

```python
import pytest

@pytest.mark.unit
def test_basic_function():
    """単体テスト"""
    pass

@pytest.mark.integration
def test_integration():
    """統合テスト"""
    pass

@pytest.mark.gpu
@pytest.mark.metal
def test_metal_backend():
    """Metal GPU テスト"""
    pass

@pytest.mark.slow
def test_long_running():
    """実行時間が長いテスト"""
    pass

@pytest.mark.property
def test_property_based():
    """プロパティベーステスト（hypothesis）"""
    pass

@pytest.mark.numerical
def test_numerical_gradient():
    """数値勾配チェック"""
    pass
```

#### テスト実行

```bash
# 全テスト
makers test

# 特定のマーカー
pytest -m unit
pytest -m integration
pytest -m "not slow"

# 特定のファイル
pytest python/tests/unit/test_tensor.py

# 特定のテスト
pytest python/tests/unit/test_tensor.py::test_add

# カバレッジレポート
makers coverage
```

#### テスト設定

- **カバレッジ閾値**: 80% 以上
- **並列実行**: CPU コア数に応じて自動
- **タイムアウト**: 300 秒
- **再現性**: シード固定（`PYTHONHASHSEED=0`）

### C++ テスト（CTest）

```bash
# C++ テストビルド
makers build

# テスト実行
makers test-cpp

# または直接 CTest
cd build
ctest --output-on-failure
```

---

## ビルド

### Python バインディングのビルド

```bash
# 開発モード（editable install）
uv pip install -e ".[dev]"

# または
python setup.py develop
```

### C++ ライブラリのビルド

#### Release ビルド

```bash
makers build
```

または手動で:

```bash
mkdir -p build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGRADFLOW_BUILD_TESTS=ON \
  -DGRADFLOW_BUILD_PYTHON_BINDINGS=ON
cmake --build . --parallel
```

#### Debug ビルド

```bash
makers build-debug
```

#### オプション

- `GRADFLOW_BUILD_TESTS`: テストをビルド（デフォルト: OFF）
- `GRADFLOW_BUILD_PYTHON_BINDINGS`: Python バインディングをビルド（デフォルト: OFF）
- `GRADFLOW_ENABLE_COVERAGE`: カバレッジ計測を有効化（デフォルト: OFF）
- `GRADFLOW_ENABLE_SANITIZER`: サニタイザーを有効化（address, undefined, thread）

---

## ドキュメント

### ドキュメント生成

```bash
# Sphinx でドキュメント生成
makers docs

# ドキュメントサーバー起動
makers docs-serve
# ブラウザで http://localhost:8000 にアクセス
```

### ドキュメントスタイル

#### Python Docstring

Google スタイルを使用:

```python
def function(arg1: int, arg2: str) -> bool:
    """関数の簡潔な説明。

    詳細な説明をここに書きます。

    Args:
        arg1: 第一引数の説明
        arg2: 第二引数の説明

    Returns:
        戻り値の説明

    Raises:
        ValueError: 無効な引数の場合

    Examples:
        >>> function(1, "test")
        True
    """
    pass
```

#### C++ Doxygen

```cpp
/**
 * @brief 関数の簡潔な説明
 *
 * 詳細な説明をここに書きます。
 *
 * @param arg1 第一引数の説明
 * @param arg2 第二引数の説明
 * @return 戻り値の説明
 * @throws std::invalid_argument 無効な引数の場合
 *
 * @example
 * @code
 * auto result = function(1, "test");
 * @endcode
 */
bool function(int arg1, const std::string& arg2);
```

---

## ワークフロー

### 一般的な開発フロー

1. **機能ブランチの作成**
   ```bash
   git checkout -b feature/your-feature
   ```

2. **コーディング**
   - TDD（Test-Driven Development）を推奨
   - 小さなコミットを頻繁に

3. **コード品質チェック**
   ```bash
   # フォーマット + リント修正 + 型チェック
   makers qa
   ```

4. **テスト実行**
   ```bash
   # Python テスト
   makers test

   # C++ テスト
   makers test-cpp
   ```

5. **コミット**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

   pre-commit フックが自動的に実行されます。

6. **Push & Pull Request**
   ```bash
   git push origin feature/your-feature
   # GitHub で Pull Request を作成
   ```

### TDD サイクル

1. **Red**: 失敗するテストを書く
   ```bash
   pytest python/tests/unit/test_new_feature.py
   # FAILED
   ```

2. **Green**: テストを通す最小限のコードを実装
   ```bash
   # コード実装
   pytest python/tests/unit/test_new_feature.py
   # PASSED
   ```

3. **Refactor**: リファクタリング
   ```bash
   # コード改善
   makers qa
   pytest
   ```

### Pull Request チェックリスト

- [ ] 全テストがパス（`makers test-all`）
- [ ] コード品質チェックがパス（`makers check`）
- [ ] カバレッジが 80% 以上（`makers coverage`）
- [ ] ドキュメントが更新されている
- [ ] Changelog が更新されている（重要な変更の場合）
- [ ] コミットメッセージが規約に従っている

### コミットメッセージ規約

Conventional Commits を使用:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type**:
- `feat`: 新機能
- `fix`: バグ修正
- `docs`: ドキュメントのみの変更
- `style`: フォーマット、セミコロンなど
- `refactor`: リファクタリング
- `perf`: パフォーマンス改善
- `test`: テストの追加・修正
- `chore`: ビルドプロセスやツールの変更

**例**:
```
feat(autograd): add support for higher-order derivatives

Implement automatic differentiation for second and higher-order
derivatives using forward-over-reverse mode.

Closes #123
```

---

## トラブルシューティング

### uv のインストールに失敗する

```bash
# 手動インストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# または pip 経由
pip install uv
```

### Pyright が型エラーを報告する

Pyright は strict モードで実行されているため、mypy より厳格です。

```bash
# 型アノテーションを追加
def function(x: int) -> int:
    return x + 1

# または型チェックを無視（推奨しない）
# type: ignore
```

### Ruff がエラーを報告する

```bash
# 自動修正
makers lint-fix

# 特定のルールを無視（pyproject.toml）
[tool.ruff.lint]
ignore = ["E501"]  # 行が長すぎる

# または行ごとに無視
# ruff: noqa: E501
```

### C++ ビルドに失敗する

```bash
# ビルドディレクトリをクリーン
makers clean-build

# 依存関係を再インストール
rm -rf build
makers build
```

### テストが失敗する

```bash
# 詳細なログを表示
pytest -vv

# 特定のテストをデバッグ
pytest --pdb python/tests/unit/test_tensor.py::test_add

# ログを表示
pytest --log-cli-level=DEBUG
```

---

## リソース

- [uv ドキュメント](https://github.com/astral-sh/uv)
- [Ruff ドキュメント](https://docs.astral.sh/ruff/)
- [Pyright ドキュメント](https://microsoft.github.io/pyright/)
- [cargo-make ドキュメント](https://github.com/sagiegurari/cargo-make)
- [pytest ドキュメント](https://docs.pytest.org/)
- [CMake ドキュメント](https://cmake.org/documentation/)

---

## サポート

質問や問題がある場合は、以下の方法でサポートを受けられます：

- GitHub Issues: バグ報告や機能リクエスト
- GitHub Discussions: 質問や議論
- Discord: リアルタイムチャット（準備中）

---

**Happy coding! 🚀**
