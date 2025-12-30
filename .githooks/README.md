# Git Hooks

このディレクトリには、GradFlow プロジェクトで使用する Git フックが含まれています。

## 概要

Git フックは、Git の特定のアクション（コミット、push など）の前後に自動的に実行されるスクリプトです。

## 使用しているフック

### pre-commit

main ブランチへの直接コミットを防ぎます。

**動作:**
- main ブランチでコミットしようとすると、エラーメッセージを表示して中断
- フィーチャーブランチでの作業を促す

### pre-push

main ブランチへの直接 push を防ぎます。

**動作:**
- main ブランチに push しようとすると、エラーメッセージを表示して中断
- Pull Request を使用した手順を案内

## セットアップ

### 自動セットアップ（推奨）

プロジェクトルートで以下のコマンドを実行：

```bash
./scripts/setup-git-hooks.sh
```

### 手動セットアップ

```bash
# Git のフックディレクトリを .githooks に設定
git config core.hooksPath .githooks

# フックに実行権限を付与
chmod +x .githooks/pre-commit
chmod +x .githooks/pre-push
```

## フックを無効化する方法

### 一時的に無効化（1回のコミット/push のみ）

```bash
# コミット時
git commit --no-verify -m "message"

# push 時
git push --no-verify
```

**⚠️ 注意**: `--no-verify` は緊急時のみ使用してください。通常はフィーチャーブランチと Pull Request を使用することを強く推奨します。

### 完全に無効化

```bash
# フック設定を解除
git config --unset core.hooksPath

# または、フックファイルを削除
rm .githooks/pre-commit
rm .githooks/pre-push
```

## ワークフロー

### 推奨される開発フロー

1. **フィーチャーブランチを作成**

```bash
git checkout -b feature/add-tensor-class
```

2. **変更を加えてコミット**

```bash
git add .
git commit -m "Add Tensor class implementation"
```

3. **ブランチを push**

```bash
git push origin feature/add-tensor-class
```

4. **Pull Request を作成**

```bash
gh pr create --title "Add Tensor class" --body "Implements basic Tensor class for Phase 1.3"
```

5. **レビュー後、GitHub でマージ**

GitHub の UI または CLI でマージ：

```bash
gh pr merge <PR番号> --squash
```

## トラブルシューティング

### フックが動作しない

```bash
# フック設定を確認
git config core.hooksPath

# 実行権限を確認
ls -la .githooks/
```

### エラーメッセージが表示されない

フックファイルの先頭が `#!/bin/bash` になっているか確認してください。

### main ブランチから抜け出せない

```bash
# 新しいブランチを作成して切り替え
git checkout -b feature/fix-something

# main ブランチの変更を持っていく
git cherry-pick <commit-hash>
```

## プロジェクトメンバーへの案内

新しくプロジェクトに参加したメンバーには、以下を実行してもらいます：

```bash
# リポジトリをクローン
git clone https://github.com/Showhey798/fullScratchLibs.git
cd fullScratchLibs

# Git フックをセットアップ
./scripts/setup-git-hooks.sh
```

## 参考資料

    - [Git Hooks 公式ドキュメント](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks)
- [GitHub Flow](https://guides.github.com/introduction/flow/)
