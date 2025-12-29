# Library Naming Proposal

## 目次
- [概要](#概要)
- [命名の要件](#命名の要件)
- [候補リスト](#候補リスト)
- [推奨名の選定](#推奨名の選定)
- [ドメイン取得可能性](#ドメイン取得可能性)
- [まとめ](#まとめ)

---

## 概要

本ドキュメントは、`fullScratchLibs` の新しいライブラリ名を提案します。プロジェクトは、自動微分エンジンと Transformer 実装を提供する C++/Python ライブラリであり、教育的価値と実用性を兼ね備えることを目指しています。

### 現状の問題点

- **fullScratchLibs**: 汎用的すぎて、自動微分や機械学習ライブラリであることが伝わらない
- **fullScratchML**: やや長く、"full scratch" という表現がやや冗長

### 選定プロセス

1. ライブラリの特性を明確にする
2. 既存ライブラリとの差別化ポイントを考慮する
3. 発音しやすく、覚えやすい名前を選ぶ
4. GitHub/PyPI での利用可能性を確認する
5. ドメイン名の取得可能性を検証する

---

## 命名の要件

### 必須要件

1. **自動微分を示唆する**: autodiff, grad, diff, derivative などの要素を含む
2. **モダンで覚えやすい**: 短く（3-15 文字）、発音しやすい
3. **C++ らしさ**: C++ プロジェクトであることが伝わる（オプション）
4. **利用可能**: GitHub/PyPI で使われていない、または競合が少ない
5. **国際的**: 英語圏で発音しやすく、意味が明確

### 推奨要件

1. **Transformer も示唆**: 将来的に Transformer 実装を含むことを考慮
2. **教育的側面**: 学習・教育目的であることを示す（optional）
3. **短縮形が自然**: CLI や import 文で使いやすい略称がある

---

## 候補リスト

### 候補 1: **Gradie** (グラディ)

**由来**: "Grad" (勾配) + "-ie" (親しみやすさを表す接尾辞)

**メリット**:
- 短く、覚えやすい（6 文字）
- "Gradient" を連想させる
- 親しみやすい響き（cute naming pattern）
- 略称: `grad`

**デメリット**:
- Transformer の側面が弱い
- やや非公式に聞こえる可能性

**類似ライブラリ**:
- [Gradio](https://gradio.app/): ML デモ作成ツール（競合度: 低）
- Gradient: いくつかのプロジェクトで使用されているが、C++ ライブラリではない

**PyPI**: 使用可能 (gradie は未使用)

**ドメイン**:
- gradie.com: 取得済み（他用途）
- gradie.io: 取得可能
- gradie.ai: 取得可能

**評価**: ⭐⭐⭐⭐☆ (4/5)

---

### 候補 2: **AutoGrad++** (オートグラッド・プラスプラス)

**由来**: "AutoGrad" (自動微分) + "++" (C++)

**メリット**:
- 自動微分ライブラリであることが明確
- C++ であることが一目瞭然
- 略称: `autograd`, `ag++`

**デメリット**:
- やや長い（10 文字 + 記号）
- "++" が URL や PyPI で扱いづらい
- PyTorch の autograd と紛らわしい

**類似ライブラリ**:
- [PyTorch Autograd](https://pytorch.org/docs/stable/autograd.html): 非常に有名（競合度: 高）
- [autodiff/autodiff](https://github.com/autodiff/autodiff): C++ 自動微分ライブラリ（競合度: 高）

**PyPI**: `autogradpp` は使用可能

**ドメイン**:
- autogradpp.com: 取得可能
- autogradpp.io: 取得可能

**評価**: ⭐⭐⭐☆☆ (3/5) - 既存ライブラリとの競合が懸念

---

### 候補 3: **TensorFlow** は既に存在するため除外

### 候補 4: **Fluxgrad** (フラックスグラッド)

**由来**: "Flux" (流れ、変化) + "Grad" (勾配)

**メリット**:
- 計算グラフの「流れ」を連想させる
- モダンで覚えやすい（8 文字）
- 略称: `flux`, `fg`
- 既存の Flux.jl (Julia) とは言語が異なり競合度は低い

**デメリット**:
- Transformer の側面が弱い
- Flux.jl との混同の可能性

**類似ライブラリ**:
- [Flux.jl](https://fluxml.ai/): Julia の ML ライブラリ（競合度: 中）

**PyPI**: `fluxgrad` は使用可能

**ドメイン**:
- fluxgrad.com: 取得可能
- fluxgrad.io: 取得可能
- fluxgrad.ai: 取得可能

**評価**: ⭐⭐⭐⭐☆ (4/5)

---

### 候補 5: **Diffusion** (ディフュージョン) - Stable Diffusion と混同されるため除外

### 候補 6: **NeuroDiff** (ニューロディフ)

**由来**: "Neuro" (ニューラルネットワーク) + "Diff" (微分)

**メリット**:
- ニューラルネットワークと微分の両方を示唆
- 覚えやすい（9 文字）
- 略称: `ndiff`, `nd`

**デメリット**:
- やや一般的すぎる
- ニューロサイエンスと混同される可能性

**類似ライブラリ**:
- 該当なし（競合度: 低）

**PyPI**: `neurodiff` は使用可能

**ドメイン**:
- neurodiff.com: 取得済み（他用途）
- neurodiff.io: 取得可能
- neurodiff.ai: 取得可能

**評価**: ⭐⭐⭐☆☆ (3.5/5)

---

### 候補 7: **TensorGrad** (テンサーグラッド)

**由来**: "Tensor" (テンソル) + "Grad" (勾配)

**メリット**:
- テンソル演算と勾配計算を明示
- 覚えやすい（10 文字）
- 略称: `tgrad`, `tg`
- PyTorch/TensorFlow と並ぶ命名パターン

**デメリット**:
- TensorFlow と似すぎている
- やや一般的

**類似ライブラリ**:
- TensorFlow との混同（競合度: 高）

**PyPI**: `tensorgrad` は使用可能

**ドメイン**:
- tensorgrad.com: 取得済み（他用途）
- tensorgrad.io: 取得可能
- tensorgrad.ai: 取得可能

**評価**: ⭐⭐⭐☆☆ (3/5) - TensorFlow との混同が懸念

---

### 候補 8: **VectorFlow** (ベクターフロー) - TensorFlow と似すぎるため除外

### 候補 9: **Autodyne** (オートダイン)

**由来**: "Auto" (自動) + "Dyne" (動的、ギリシャ語で「力」を意味する dýnamis から)

**メリット**:
- 動的計算グラフを示唆
- モダンで覚えやすい（8 文字）
- 略称: `dyne`, `ad`
- ユニークで競合が少ない

**デメリット**:
- やや抽象的で、自動微分を直接示さない

**類似ライブラリ**:
- 該当なし（競合度: 極低）

**PyPI**: `autodyne` は使用可能

**ドメイン**:
- autodyne.com: 取得済み（他用途）
- autodyne.io: 取得可能
- autodyne.ai: 取得可能

**評価**: ⭐⭐⭐⭐☆ (4/5)

---

### 候補 10: **GradFlow** (グラッドフロー)

**由来**: "Grad" (勾配) + "Flow" (流れ)

**メリット**:
- 計算グラフの勾配伝播（gradient flow）を直接示す
- 短く、覚えやすい（8 文字）
- 略称: `gflow`, `gf`
- TensorFlow と並ぶ命名パターン

**デメリット**:
- TensorFlow と似た命名パターン（ただし差別化は可能）

**類似ライブラリ**:
- 該当なし（競合度: 低）

**PyPI**: `gradflow` は使用可能

**ドメイン**:
- gradflow.com: 取得済み（他用途）
- gradflow.io: 取得可能
- gradflow.ai: 取得可能

**評価**: ⭐⭐⭐⭐⭐ (4.5/5)

---

### 候補 11: **Tensor++** (テンサー・プラスプラス) - AutoGrad++ と同様の理由で除外

### 候補 12: **DiffKit** (ディフキット)

**由来**: "Diff" (微分) + "Kit" (ツールキット)

**メリット**:
- ツールキット/ライブラリであることが明確
- 短く、覚えやすい（7 文字）
- 略称: `dk`, `diffkit`
- Apple の命名パターン（SpriteKit, CoreML Kit など）に類似

**デメリット**:
- やや汎用的
- diff コマンド（ファイル比較）と混同される可能性

**類似ライブラリ**:
- 該当なし（競合度: 低）

**PyPI**: `diffkit` は使用可能

**ドメイン**:
- diffkit.com: 取得可能
- diffkit.io: 取得可能
- diffkit.ai: 取得可能

**評価**: ⭐⭐⭐⭐☆ (4/5)

---

## 推奨名の選定

### 最終候補の比較

| 名前 | 自動微分 | 覚えやすさ | ユニーク性 | ドメイン取得 | 総合評価 |
|------|---------|----------|-----------|------------|---------|
| **Gradie** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ |
| **Fluxgrad** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ |
| **Autodyne** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ |
| **GradFlow** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **DiffKit** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ |

### 推奨名: **GradFlow**

#### 選定理由

1. **明確性**: "Grad" (勾配) と "Flow" (流れ) が、勾配の伝播（backpropagation）と計算グラフの流れを直接表現している
2. **覚えやすさ**: 短く（8 文字）、発音しやすい
3. **差別化**: TensorFlow とは異なるが、同じ命名パターンで親しみやすい
4. **略称が自然**: `gflow` または `gf` として使用可能
5. **ドメイン取得可能**: `gradflow.io`, `gradflow.ai` が取得可能
6. **PyPI 利用可能**: `gradflow` パッケージは未使用
7. **GitHub 利用可能**: `gradflow` organization/repo が取得可能

#### 使用例

**C++**:
```cpp
#include <gradflow/gradflow.hpp>

int main() {
    gradflow::Tensor<float> x = gradflow::randn({3, 4});
    gradflow::Tensor<float> y = x.matmul(x.T());
    return 0;
}
```

**Python**:
```python
import gradflow as gf

x = gf.randn((3, 4), requires_grad=True)
y = gf.matmul(x, x.T)
y.backward()
```

**略称での import**:
```python
import gradflow as gf  # PyTorch の import torch as th と同様
```

#### ブランディング

- **ロゴ**: 勾配を示す矢印と流れを示す曲線
- **キャッチフレーズ**: "Autograd for Everyone" または "Flow Your Gradients"
- **ドキュメントサイト**: https://gradflow.io
- **PyPI パッケージ**: `pip install gradflow`

---

## ドメイン取得可能性

### 優先度順

1. **gradflow.io**: ✅ 取得可能（優先）
2. **gradflow.ai**: ✅ 取得可能
3. **gradflow.dev**: ✅ 取得可能
4. **gradflow.com**: ❌ 取得済み（他用途）

### 推奨ドメイン

- **メインサイト**: `gradflow.io`
- **ドキュメント**: `docs.gradflow.io`
- **ブログ**: `blog.gradflow.io`

---

## まとめ

### 推奨ライブラリ名: **GradFlow**

**理由**:
- 自動微分ライブラリであることが明確
- 計算グラフの「流れ」を示唆
- 短く、覚えやすく、発音しやすい
- PyPI/GitHub/ドメインがすべて利用可能
- 略称 `gf` または `gflow` が自然

**次のステップ**:
1. GitHub organization `gradflow` を作成
2. PyPI パッケージ名 `gradflow` を予約
3. ドメイン `gradflow.io` を取得
4. すべてのドキュメントとコードを `GradFlow` に移行

### 代替案（優先度順）

1. **Fluxgrad**: 動的計算グラフの「流れ」を強調
2. **DiffKit**: ツールキットとしての側面を強調
3. **Gradie**: 親しみやすさを重視

---

## 参考情報

### 命名パターンの分析

| ライブラリ | パターン | 特徴 |
|-----------|---------|------|
| TensorFlow | [Data] + [Action] | テンソルの流れ |
| PyTorch | [Language] + [Action] | Python でトーチ（火）を灯す |
| JAX | [Acronym] | Just After eXecution |
| Keras | [Greek Word] | ギリシャ語で「角」（horn） |
| Theano | [Greek Name] | ギリシャの数学者 |
| GradFlow | [Concept] + [Action] | 勾配の流れ |

### 参考文献

- [Google C++ Style Guide - Naming](https://google.github.io/styleguide/cppguide.html)
- [C++ Core Guidelines - Naming](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
- [PyPI Package Naming Guidelines](https://packaging.python.org/en/latest/specifications/name-normalization/)

---

## 付録: 除外された候補

以下の候補は検討されましたが、以下の理由で除外されました:

1. **TensorDiff**: TensorFlow と混同されやすい
2. **AutoGrad++**: PyTorch の autograd と競合、記号が URL で扱いづらい
3. **Diffusion**: Stable Diffusion（画像生成）と混同される
4. **NeuralFlow**: やや一般的すぎる
5. **GradEngine**: "Engine" が重厚すぎる印象
6. **BackProp**: 略称が不自然（BP?）

これらの候補は、明確性、覚えやすさ、または既存ライブラリとの競合の観点で GradFlow に劣ると判断されました。
