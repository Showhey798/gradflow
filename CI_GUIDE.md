# CI/CD ガイド

このドキュメントは、GradFlow プロジェクトの CI/CD システムの使い方と、ローカルでの検証方法を説明します。

## 哲学

**「ローカルと CI の完全な同期」** - ローカルで実行できる検証は、CI でも同じように実行されます。プッシュ前にローカルで検証することで、CI の失敗を防ぎます。

## クイックスタート

### Pre-commit のセットアップ

コミット前に自動的にコードをチェックします。

```bash
# pre-commit のインストール
pip install pre-commit

# フックのインストール
pre-commit install

# 全ファイルに対して手動実行
pre-commit run --all-files
```

### ローカルでの CI 検証

CI と同じ検証をローカルで実行できます。

```bash
# 全ての CI チェックを実行
./scripts/ci-verify.sh

# 個別のチェックを実行
./scripts/ci-format-check.sh    # C++ フォーマットチェック
./scripts/ci-format-apply.sh    # C++ フォーマット適用
./scripts/ci-lint-cpp.sh        # C++ 静的解析 (clang-tidy)
./scripts/ci-lint-python.sh     # Python 品質チェック (ruff, pyright)
./scripts/ci-test-cpp.sh        # C++ ビルド & テスト
```

## CI スクリプト詳細

### `ci-format-check.sh`

C++ コードが clang-format に準拠しているかチェックします。

```bash
./scripts/ci-format-check.sh
```

**要件:**
- clang-format (version 15 以降推奨)

### `ci-format-apply.sh`

C++ コードに clang-format を適用します。

```bash
./scripts/ci-format-apply.sh
```

### `ci-lint-cpp.sh`

clang-tidy を使用して C++ コードの静的解析を行います。

```bash
# 事前にビルドディレクトリを生成する必要があります
conan install . --output-folder=build --build=missing -s build_type=Debug
cmake --preset conan-debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# 静的解析を実行
./scripts/ci-lint-cpp.sh
```

**要件:**
- clang-tidy (version 15 以降推奨)
- compile_commands.json (CMake で生成)

### `ci-lint-python.sh`

Python コードの品質チェックを行います。

```bash
./scripts/ci-lint-python.sh
```

**チェック内容:**
- ruff format: コードフォーマットチェック
- ruff check: リンティング
- pyright: 型チェック (オプション)

**要件:**
- ruff
- pyright (オプション)

```bash
pip install ruff pyright
```

### `ci-test-cpp.sh`

C++ のビルドとテストを実行します。

```bash
# デフォルト (Release ビルド)
./scripts/ci-test-cpp.sh

# Debug ビルド
./scripts/ci-test-cpp.sh conan-debug
```

### `ci-verify.sh`

全ての CI チェックを一括で実行します。

```bash
./scripts/ci-verify.sh
```

これは以下を順次実行します：
1. C++ フォーマットチェック
2. Python 品質チェック
3. C++ ビルド & テスト

## GitHub Actions の構成

### ワークフロー一覧

1. **ci.yml** - メイン CI パイプライン
   - コード品質チェック (clang-format)
   - マルチプラットフォームビルド & テスト
   - カバレッジ測定
   - Sanitizer (AddressSanitizer, UndefinedBehaviorSanitizer)
   - ドキュメントビルド

2. **code_quality.yml** - コード品質チェック
   - Python 品質 (ruff, pyright)
   - C++ フォーマット (clang-format)
   - C++ 静的解析 (clang-tidy)
   - その他の静的解析ツール (cppcheck, IWYU, 複雑度解析等は無効化中)

3. **python_tests.yml** - Python バインディングテスト
   - 現在は無効化中 (Python バインディング実装後に有効化)

### セキュリティのベストプラクティス

全ての GitHub Actions は以下のセキュリティ対策を実施しています：

1. **最小権限の原則**
   - トップレベルで `permissions: contents: read` を設定
   - ジョブごとに必要最小限の権限のみを付与

2. **アクションの SHA 固定**
   - 全てのアクションを不変の commit SHA でピン留め
   - タグ (@v4 等) は使用せず、SHA を使用
   - 例: `actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5 # v4`

3. **キャッシュ戦略**
   - Python 依存関係のキャッシュ (`actions/setup-python` の `cache: 'pip'`)
   - Conan パッケージのキャッシュ (`actions/cache`)
   - ビルド時間の大幅な短縮

## ローカル開発のワークフロー

### 1. 開発開始

```bash
# リポジトリをクローン
git clone <repository-url>
cd gradflow

# pre-commit をセットアップ
pip install pre-commit
pre-commit install
```

### 2. コード変更

```bash
# コードを編集...

# フォーマットを適用 (オプション)
./scripts/ci-format-apply.sh
```

### 3. コミット前の検証

```bash
# pre-commit が自動実行されます
git add .
git commit -m "feat: Add new feature"

# または、手動で全チェックを実行
./scripts/ci-verify.sh
```

### 4. プッシュ

```bash
git push origin feature-branch
```

ローカルで `ci-verify.sh` が成功していれば、CI も 99% 成功します。

## トラブルシューティング

### Q: clang-format が見つからない

```bash
# Ubuntu/Debian
sudo apt-get install clang-format-15

# macOS
brew install clang-format

# シンボリックリンクを作成 (オプション)
sudo ln -sf /usr/bin/clang-format-15 /usr/bin/clang-format
```

### Q: clang-tidy が "compile_commands.json not found" エラー

```bash
# ビルドディレクトリを生成
conan install . --output-folder=build --build=missing -s build_type=Debug
cmake --preset conan-debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

### Q: pre-commit が遅い

以下の設定で特定のフックをスキップできます：

```bash
# 特定のフックをスキップ
SKIP=pyright pre-commit run --all-files

# または、.pre-commit-config.yaml で pyright をコメントアウト
```

### Q: CI でキャッシュが効かない

キャッシュキーは以下に依存します：
- `conanfile.py` の内容 (Conan 依存関係)
- OS とコンパイラのバージョン

これらを変更した場合、キャッシュは無効化されます。

## 参考資料

- [GitHub Actions Security Best Practices](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Clang-Format](https://clang.llvm.org/docs/ClangFormat.html)
- [Clang-Tidy](https://clang.llvm.org/extra/clang-tidy/)
- [Ruff](https://docs.astral.sh/ruff/)
