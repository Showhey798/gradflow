#!/bin/bash

# GradFlow プロジェクトクリーンアップスクリプト
# このスクリプトは、ロードマップに含まれていない古いファイルを削除します

set -e  # エラーで停止

echo "========================================="
echo "GradFlow プロジェクトクリーンアップ"
echo "========================================="
echo ""
echo "このスクリプトは以下のファイルを削除します："
echo "  - 古いドキュメント（4ファイル）"
echo "  - 古い名前を含むファイル（FullScratch関連）"
echo "  - 設計にない初期実装ファイル（Matrix/Vector等）"
echo "  - 古いテストファイル（7ファイル）"
echo "  - 古い例（2ファイル）"
echo "  - Python バインディングの古いファイル（8ファイル）"
echo "  - その他（conanfile.txt）"
echo ""
read -p "削除を実行しますか？ (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "キャンセルしました。"
    exit 1
fi

echo ""
echo "削除を開始します..."
echo ""

# カウンター
deleted_count=0

# 1. 古いドキュメント
echo "[1/8] 古いドキュメントを削除..."
for file in docs/BUILD.md docs/CONTRIBUTING.md PROJECT_STATUS.md QUICKSTART.md; do
    if [ -f "$file" ]; then
        rm -f "$file"
        echo "  ✓ 削除: $file"
        ((deleted_count++))
    else
        echo "  - スキップ（存在しない）: $file"
    fi
done

# 2. 古い名前を含むファイル
echo ""
echo "[2/8] 古い名前を含むファイルを削除..."
for file in cmake/FullScratchMLConfig.cmake.in include/gradflow/fullscratch.hpp; do
    if [ -f "$file" ]; then
        rm -f "$file"
        echo "  ✓ 削除: $file"
        ((deleted_count++))
    else
        echo "  - スキップ（存在しない）: $file"
    fi
done

if [ -d "python/fullscratch/" ]; then
    rm -rf "python/fullscratch/"
    echo "  ✓ 削除: python/fullscratch/"
    ((deleted_count++))
else
    echo "  - スキップ（存在しない）: python/fullscratch/"
fi

# 3. 設計にない初期実装ファイル
echo ""
echo "[3/8] 設計にない初期実装ファイルを削除..."
for file in \
    include/gradflow/matrix.hpp \
    include/gradflow/vector.hpp \
    include/gradflow/activation.hpp \
    include/gradflow/data_loader.hpp \
    include/gradflow/loss.hpp \
    include/gradflow/neural_network.hpp \
    include/gradflow/optimizer.hpp; do
    if [ -f "$file" ]; then
        rm -f "$file"
        echo "  ✓ 削除: $file"
        ((deleted_count++))
    else
        echo "  - スキップ（存在しない）: $file"
    fi
done

# 4. 古いテストファイル
echo ""
echo "[4/8] 古いテストファイルを削除..."
for file in \
    tests/test_vector.cpp \
    tests/test_matrix.cpp \
    tests/test_activation.cpp \
    tests/test_data_loader.cpp \
    tests/test_loss.cpp \
    tests/test_neural_network.cpp \
    tests/test_integration.cpp; do
    if [ -f "$file" ]; then
        rm -f "$file"
        echo "  ✓ 削除: $file"
        ((deleted_count++))
    else
        echo "  - スキップ（存在しない）: $file"
    fi
done

# 5. 古い例
echo ""
echo "[5/8] 古い例を削除..."
for file in examples/matrix_operations.cpp examples/simple_neural_network.cpp; do
    if [ -f "$file" ]; then
        rm -f "$file"
        echo "  ✓ 削除: $file"
        ((deleted_count++))
    else
        echo "  - スキップ（存在しない）: $file"
    fi
done

# 6. Python バインディングの古いファイル
echo ""
echo "[6/8] Python バインディングの古いファイルを削除..."
for file in \
    python/activation_bindings.cpp \
    python/loss_bindings.cpp \
    python/matrix_bindings.cpp \
    python/neural_network_bindings.cpp \
    python/optimizer_bindings.cpp \
    python/vector_bindings.cpp \
    python/bindings.cpp \
    python/setup.py; do
    if [ -f "$file" ]; then
        rm -f "$file"
        echo "  ✓ 削除: $file"
        ((deleted_count++))
    else
        echo "  - スキップ（存在しない）: $file"
    fi
done

# 7. その他
echo ""
echo "[7/8] その他のファイルを削除..."
if [ -f "conanfile.txt" ]; then
    rm -f "conanfile.txt"
    echo "  ✓ 削除: conanfile.txt"
    ((deleted_count++))
else
    echo "  - スキップ（存在しない）: conanfile.txt"
fi

# 8. 空ディレクトリの確認
echo ""
echo "[8/8] ディレクトリ構造を確認..."
mkdir -p include/gradflow/autograd
mkdir -p include/gradflow/nn
mkdir -p include/gradflow/optim
mkdir -p src/autograd/cpu
mkdir -p src/autograd/metal
mkdir -p src/autograd/cuda
mkdir -p src/nn
echo "  ✓ 必要なディレクトリ構造を作成"

echo ""
echo "========================================="
echo "クリーンアップ完了"
echo "========================================="
echo ""
echo "削除されたファイル数: $deleted_count"
echo ""
echo "次のステップ："
echo "  1. git status で削除を確認"
echo "  2. git add -A で削除をステージング"
echo "  3. git commit でコミット"
echo "  4. git push でリモートに push"
echo ""
