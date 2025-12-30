# ローカル CI 実行ガイド

このドキュメントでは、GitHub Actions で実行される CI チェックをローカルで実行する方法を説明します。

## 前提条件

### 必須ツール

1. **C++ 開発ツール**
   ```bash
   # macOS
   xcode-select --install

   # Ubuntu/Debian
   sudo apt-get install build-essential cmake ninja-build

   # Fedora/RHEL
   sudo dnf install gcc-c++ cmake ninja-build
   ```

2. **clang-format** (C++ フォーマット)
   ```bash
   # macOS
   brew install clang-format

   # Ubuntu/Debian
   sudo apt-get install clang-format-15

   # Fedora/RHEL
   sudo dnf install clang-tools-extra
   ```

3. **cmake-format** (CMake フォーマット)
   ```bash
   pip install cmakelang
   # または
   python3 -m pip install --user cmakelang
   ```

4. **Python ツール** (Python バインディングを使用する場合)
   ```bash
   pip install ruff pyright pytest pytest-cov
   ```

5. **Conan** (依存関係管理)
   ```bash
   pip install conan==2.0.17
   conan profile detect --force
   ```

### オプションツール

1. **clang-tidy** (C++ 静的解析)
   ```bash
   # macOS
   brew install llvm
   # clang-tidy は llvm に含まれます

   # Ubuntu/Debian
   sudo apt-get install clang-tidy-15
   ```

## 使用方法

### 全ての CI チェックを実行

```bash
./scripts/run-ci-local.sh
```

このコマンドは以下のチェックを順番に実行します：
1. CMake フォーマットチェック
2. C++ フォーマットチェック
3. Python フォーマットチェック（存在する場合）
4. C++ リント (clang-tidy)
5. ビルド
6. C++ テスト
7. Python テスト（存在する場合）

### クイックモード（リントをスキップ）

clang-tidy は時間がかかるため、クイックチェック用にスキップできます：

```bash
./scripts/run-ci-local.sh --quick
```

### テストをスキップ

ビルドのみ確認したい場合：

```bash
./scripts/run-ci-local.sh --skip-tests
```

### オプションの組み合わせ

```bash
./scripts/run-ci-local.sh --quick --skip-tests
```

## 個別スクリプト

各チェックを個別に実行することもできます：

### フォーマットチェック

```bash
# CMake フォーマット
./scripts/ci-format-cmake.sh

# C++ フォーマット
./scripts/ci-format-check.sh

# Python フォーマット
./scripts/ci-lint-python.sh
```

### フォーマット適用

```bash
# C++ フォーマットを自動適用
./scripts/ci-format-apply.sh

# CMake フォーマットを自動適用
find . -name 'CMakeLists.txt' -o -name '*.cmake' | xargs cmake-format -i

# Python フォーマットを自動適用
ruff format python/
```

### リント

```bash
# C++ リント (clang-tidy)
./scripts/ci-lint-cpp.sh

# Python リント
./scripts/ci-lint-python.sh
```

### ビルドとテスト

```bash
# ビルド
./scripts/build.sh

# C++ テスト
./scripts/ci-test-cpp.sh

# Python テスト
./scripts/ci-test-python.sh
```

## CI との差分

ローカルで実行する場合、以下の点が GitHub Actions と異なります：

1. **環境の違い**: ローカルの OS、コンパイラバージョン、依存関係のバージョンが CI と異なる可能性があります
2. **並列実行**: CI は複数のジョブを並列実行しますが、ローカルスクリプトは順次実行します
3. **クリーンな環境**: CI は毎回クリーンな環境で実行されますが、ローカルはビルドキャッシュが残る可能性があります

完全にクリーンな状態でテストする場合は、ビルドディレクトリを削除してから実行してください：

```bash
rm -rf build
./scripts/run-ci-local.sh
```

## トラブルシューティング

### cmake-format not found

```bash
pip install cmakelang
# または
python3 -m pip install --user --break-system-packages cmakelang
```

インストール後、PATH が通っていることを確認：

```bash
which cmake-format
```

### clang-format not found

macOS の場合：

```bash
brew install clang-format
```

Linux の場合：

```bash
# Ubuntu/Debian
sudo apt-get install clang-format-15
sudo ln -sf /usr/bin/clang-format-15 /usr/bin/clang-format

# Fedora/RHEL
sudo dnf install clang-tools-extra
```

### clang-tidy not found

clang-tidy は必須ではありません。`--quick` オプションを使用してスキップできます：

```bash
./scripts/run-ci-local.sh --quick
```

インストールする場合：

```bash
# macOS
brew install llvm

# Ubuntu/Debian
sudo apt-get install clang-tidy-15
sudo ln -sf /usr/bin/clang-tidy-15 /usr/bin/clang-tidy
```

### Conan の問題

Conan プロファイルが正しく設定されていない場合：

```bash
conan profile detect --force
```

依存関係のキャッシュをクリア：

```bash
rm -rf ~/.conan2
conan profile detect --force
```

## Git フック（推奨）

コミット前に自動的にチェックを実行するには、Git フックをセットアップします：

```bash
./scripts/setup-git-hooks.sh
```

これにより、`git commit` 時に自動的にフォーマットチェックとリントが実行されます。

## まとめ

- **プッシュ前**: `./scripts/run-ci-local.sh` を実行
- **クイックチェック**: `./scripts/run-ci-local.sh --quick`
- **フォーマット修正**: `./scripts/ci-format-apply.sh`
- **Git フック**: `./scripts/setup-git-hooks.sh` でセットアップ

これらのツールを活用することで、CI の失敗を事前に防ぎ、効率的な開発ができます。
