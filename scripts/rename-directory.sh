#!/bin/bash

# GradFlow - ディレクトリ名変更スクリプト
# fullScratchLibs → gradflow

set -e

echo "========================================="
echo "ディレクトリ名を gradflow に変更"
echo "========================================="
echo ""

# 現在のディレクトリを確認
CURRENT_DIR=$(pwd)
DIR_NAME=$(basename "$CURRENT_DIR")

if [ "$DIR_NAME" != "fullScratchLibs" ]; then
    echo "⚠️  警告: 現在のディレクトリは fullScratchLibs ではありません"
    echo "現在のディレクトリ: $DIR_NAME"
    echo ""
    read -p "それでも続行しますか？ (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "キャンセルしました。"
        exit 1
    fi
fi

echo "現在のディレクトリ: $CURRENT_DIR"
echo ""
echo "このスクリプトは以下の操作を行います："
echo "  1. 親ディレクトリに移動"
echo "  2. ディレクトリ名を gradflow に変更"
echo "  3. 新しいディレクトリに移動"
echo ""
read -p "実行しますか？ (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "キャンセルしました。"
    exit 1
fi

echo ""
echo "ディレクトリ名を変更中..."

# 親ディレクトリのパスを取得
PARENT_DIR=$(dirname "$CURRENT_DIR")
NEW_PATH="$PARENT_DIR/gradflow"

# 既に gradflow が存在するか確認
if [ -d "$NEW_PATH" ]; then
    echo "❌ エラー: $NEW_PATH は既に存在します"
    exit 1
fi

# 親ディレクトリに移動して名前変更
cd "$PARENT_DIR"
mv "$DIR_NAME" gradflow

if [ $? -eq 0 ]; then
    echo "  ✓ ディレクトリ名を変更しました"
    echo ""
    echo "========================================="
    echo "変更完了"
    echo "========================================="
    echo ""
    echo "新しいパス: $NEW_PATH"
    echo ""
    echo "次のコマンドで新しいディレクトリに移動してください："
    echo "  cd $NEW_PATH"
    echo ""
else
    echo "❌ エラー: ディレクトリ名の変更に失敗しました"
    exit 1
fi
