#!/bin/bash

# GradFlow - Git Hooks セットアップスクリプト
# main ブランチへの直接コミット/push を防ぐためのフックを設定します

set -e  # エラーで停止

echo "========================================="
echo "Git Hooks セットアップ"
echo "========================================="
echo ""

# プロジェクトルートディレクトリを取得
PROJECT_ROOT=$(git rev-parse --show-toplevel)

if [ ! -d "$PROJECT_ROOT/.githooks" ]; then
    echo "❌ エラー: .githooks ディレクトリが見つかりません"
    echo "このスクリプトはプロジェクトルートから実行してください"
    exit 1
fi

echo "プロジェクトルート: $PROJECT_ROOT"
echo ""

# Git のフックディレクトリを .githooks に設定
echo "[1/3] Git のフックディレクトリを設定中..."
git config core.hooksPath .githooks
echo "  ✓ core.hooksPath = .githooks"
echo ""

# フックファイルに実行権限を付与
echo "[2/3] フックファイルに実行権限を付与中..."
if [ -f "$PROJECT_ROOT/.githooks/pre-commit" ]; then
    chmod +x "$PROJECT_ROOT/.githooks/pre-commit"
    echo "  ✓ pre-commit"
else
    echo "  ⚠ pre-commit が見つかりません"
fi

if [ -f "$PROJECT_ROOT/.githooks/pre-push" ]; then
    chmod +x "$PROJECT_ROOT/.githooks/pre-push"
    echo "  ✓ pre-push"
else
    echo "  ⚠ pre-push が見つかりません"
fi
echo ""

# 設定を確認
echo "[3/3] 設定を確認中..."
HOOKS_PATH=$(git config core.hooksPath)
if [ "$HOOKS_PATH" = ".githooks" ]; then
    echo "  ✓ Git フックが正しく設定されました"
else
    echo "  ❌ 設定が正しくありません: $HOOKS_PATH"
    exit 1
fi
echo ""

echo "========================================="
echo "セットアップ完了"
echo "========================================="
echo ""
echo "✅ main ブランチへの直接コミット/push が禁止されました"
echo ""
echo "有効化されたフック："
echo "  - pre-commit: main ブランチでのコミットを防ぐ"
echo "  - pre-push: main ブランチへの push を防ぐ"
echo ""
echo "推奨される開発フロー："
echo "  1. git checkout -b feature/your-feature"
echo "  2. git commit -m \"your changes\""
echo "  3. git push origin feature/your-feature"
echo "  4. gh pr create"
echo ""
echo "詳細は .githooks/README.md を参照してください。"
echo ""
