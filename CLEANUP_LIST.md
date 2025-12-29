# GradFlow クリーンアップリスト

このドキュメントは、プロジェクトのロードマップと実装計画に含まれていない、削除すべきファイルとディレクトリをリストアップしています。

## 削除対象ファイル・ディレクトリ

### 1. 古いドキュメント（設計にない）

```
docs/BUILD.md
docs/CONTRIBUTING.md
PROJECT_STATUS.md
QUICKSTART.md
```

**理由**: これらのドキュメントは設計ドキュメントに含まれておらず、README.md で十分カバーされています。

---

### 2. 古い名前を含むファイル（FullScratch → GradFlow）

```
cmake/FullScratchMLConfig.cmake.in
include/gradflow/fullscratch.hpp
python/fullscratch/
python/fullscratch/__init__.py
```

**理由**: ライブラリ名を FullScratch から GradFlow に変更したため、古い名前を含むファイルは不要です。

---

### 3. 設計にない初期実装ファイル（include/gradflow/）

現在の構造では、ヘッダーファイルが `include/gradflow/` 直下にありますが、設計では以下のサブディレクトリ構造を使用します：
- `include/gradflow/autograd/` - 自動微分関連
- `include/gradflow/nn/` - ニューラルネットワークコンポーネント
- `include/gradflow/optim/` - 最適化アルゴリズム

**削除対象:**
```
include/gradflow/matrix.hpp          # Tensor に置き換え
include/gradflow/vector.hpp          # Tensor に置き換え
include/gradflow/activation.hpp      # → autograd/ops/activation.hpp
include/gradflow/data_loader.hpp     # 設計に含まれていない
include/gradflow/loss.hpp            # → autograd/ops/loss.hpp
include/gradflow/neural_network.hpp  # nn/ ディレクトリに分割
include/gradflow/optimizer.hpp       # → optim/optimizer.hpp
```

**理由**: 設計ドキュメントでは、Tensor ベースの実装と明確なディレクトリ構造を定義しています。Matrix/Vector は初期のプロトタイプです。

---

### 4. 古いテストファイル（tests/）

```
tests/test_vector.cpp             # vector.hpp が不要
tests/test_matrix.cpp             # matrix.hpp が不要
tests/test_activation.cpp         # 新構造: test_activation_grad.cpp
tests/test_data_loader.cpp        # data_loader.hpp が不要
tests/test_loss.cpp               # 新構造: test_loss_grad.cpp
tests/test_neural_network.cpp     # 古い構造
tests/test_integration.cpp        # プレースホルダーのみ
```

**理由**: 削除するヘッダーファイルに対応するテストも不要です。新しい設計では Phase ごとに適切なテストが定義されています。

**注意**: `tests/test_optimizer.cpp` は残しますが、内容は Phase 2.5 で再実装が必要です。

---

### 5. 古い例（examples/）

```
examples/matrix_operations.cpp
examples/simple_neural_network.cpp
```

**理由**: これらは初期のプロトタイプ例です。Phase 6.1 で適切なエンドツーエンド例（MNIST、Transformer翻訳、言語モデル）を実装します。

---

### 6. Python バインディングの古いファイル（python/）

```
python/activation_bindings.cpp
python/loss_bindings.cpp
python/matrix_bindings.cpp
python/neural_network_bindings.cpp
python/optimizer_bindings.cpp
python/vector_bindings.cpp
python/bindings.cpp
python/setup.py
```

**理由**:
- 古い Matrix/Vector ベースの実装に対応
- 設計では nanobind を使用した統合的なバインディングを Phase 6.4 で実装
- `python/pyproject.toml` は残しますが、内容の更新が必要

---

### 7. その他の削除候補

```
conanfile.txt
```

**理由**: `conanfile.py` を使用するため、`conanfile.txt` は不要です。

---

## 保持するが更新が必要なファイル

以下のファイルは削除せず、内容の更新が必要です：

```
CMakeLists.txt              # プロジェクト名、構造に合わせて更新
examples/CMakeLists.txt     # 新しい例に合わせて更新
python/CMakeLists.txt       # nanobind に変更
python/pyproject.toml       # パッケージ名を gradflow に変更
tests/CMakeLists.txt        # 新しいテスト構造に合わせて更新
tests/test_optimizer.cpp    # Phase 2.5 で再実装
scripts/build.sh            # プロジェクト名に合わせて更新
cmake/FullScratchMLConfig.cmake.in  # GradFlowConfig.cmake.in に名前変更と更新
```

---

## 保持する空ディレクトリ

以下のディレクトリは現在空ですが、将来の実装で使用されるため保持します：

```
src/                 # Phase 1 から使用開始
benchmarks/          # Phase 6.2 で使用
```

---

## 削除コマンド

**⚠️ 注意: 以下のコマンドを実行する前に、必ずバックアップを取るか、Git でコミットしてください！**

```bash
# ドキュメント
rm -f docs/BUILD.md
rm -f docs/CONTRIBUTING.md
rm -f PROJECT_STATUS.md
rm -f QUICKSTART.md

# 古い名前を含むファイル
rm -f cmake/FullScratchMLConfig.cmake.in
rm -f include/gradflow/fullscratch.hpp
rm -rf python/fullscratch/

# include/gradflow/ 直下の古いファイル
rm -f include/gradflow/matrix.hpp
rm -f include/gradflow/vector.hpp
rm -f include/gradflow/activation.hpp
rm -f include/gradflow/data_loader.hpp
rm -f include/gradflow/loss.hpp
rm -f include/gradflow/neural_network.hpp
rm -f include/gradflow/optimizer.hpp

# 古いテストファイル
rm -f tests/test_vector.cpp
rm -f tests/test_matrix.cpp
rm -f tests/test_activation.cpp
rm -f tests/test_data_loader.cpp
rm -f tests/test_loss.cpp
rm -f tests/test_neural_network.cpp
rm -f tests/test_integration.cpp

# 古い例
rm -f examples/matrix_operations.cpp
rm -f examples/simple_neural_network.cpp

# Python バインディングの古いファイル
rm -f python/activation_bindings.cpp
rm -f python/loss_bindings.cpp
rm -f python/matrix_bindings.cpp
rm -f python/neural_network_bindings.cpp
rm -f python/optimizer_bindings.cpp
rm -f python/vector_bindings.cpp
rm -f python/bindings.cpp
rm -f python/setup.py

# その他
rm -f conanfile.txt
```

---

## 削除後の状態

削除後、以下のファイルとディレクトリのみが残ります：

```
GradFlow/
├── .clang-format
├── .clang-tidy
├── .conan/
├── .editorconfig
├── .github/              # CI/CD workflows（更新予定）
├── CMakeLists.txt        # 更新必要
├── LICENSE
├── README.md
├── benchmarks/           # 空（Phase 6.2 で使用）
├── cmake/                # 空または削除後に GradFlowConfig.cmake.in を追加
├── conanfile.py
├── docs/
│   ├── API_DESIGN.md
│   ├── ARCHITECTURE.md
│   ├── LIBRARY_NAMING.md
│   ├── ROADMAP.md
│   └── TECHNICAL_DECISIONS.md
├── examples/
│   └── CMakeLists.txt    # 更新必要
├── include/
│   └── gradflow/         # 空（Phase 1 から実装開始）
│       ├── autograd/     # Phase 1-3 で実装
│       ├── nn/           # Phase 4-5 で実装
│       └── optim/        # Phase 2.5 で実装
├── python/
│   ├── CMakeLists.txt    # 更新必要
│   └── pyproject.toml    # 更新必要
├── scripts/
│   └── build.sh          # 更新必要
├── src/                  # 空（Phase 1 から使用）
│   ├── autograd/
│   │   ├── cpu/
│   │   ├── metal/
│   │   └── cuda/
│   └── nn/
└── tests/
    ├── CMakeLists.txt    # 更新必要
    └── test_optimizer.cpp # プレースホルダー（Phase 2.5 で再実装）
```

---

## 次のステップ

1. **バックアップまたは Git コミット**: 削除前に現在の状態を保存
2. **削除実行**: 上記の削除コマンドを実行
3. **Phase 1 実装開始**: Issue #1 から順番に実装を開始
4. **CI/CD 更新**: `.github/workflows/` 内のファイルをプロジェクト構造に合わせて更新

---

## 確認事項

削除を実行する前に、以下を確認してください：

- [ ] 現在の作業が Git でコミットされている
- [ ] 削除対象のファイルに重要な実装が含まれていないか確認
- [ ] 削除コマンドのパスが正しいか確認
- [ ] バックアップが取られているか確認

削除後、クリーンな状態から Phase 1 の実装を開始できます。
