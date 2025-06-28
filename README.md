# AdamW-mini-SF
Lightweight optimizer with ScheduleFree &amp; AMP support

**Lightweight, schedule-free optimizer based on AdamW — with automatic learning rate adjustment and AMP support.**

This optimizer extends [Adam-mini](https://github.com/zyushun/Adam-mini) with:

- 🚀 **Memory-efficient state**: keeps moments (`m`, `v`) in low-precision (e.g., `float16` / `bfloat16`)
- 🧠 **Schedule-Free learning rate adaptation**: adjusts `lr` dynamically using smoothed gradient norms (no schedulers needed)
- 🛡️ **Decoupled weight decay**: follows AdamW-style decay, separate from gradients
- ⚙️ **AMP/mixed-precision support**: detects parameter dtypes for seamless integration with `torch.amp` or custom precision

## Installation

Simply copy the `adamw_mini_sf.py` file into your project.

## Usage

```python
from adamw_mini_sf import AdamWminiScheduleFree
optimizer = AdamWminiScheduleFree(model.parameters(), lr=1e-3)
```

License
Apache License 2.0 — see LICENSE for details.

Built with 🤖 GitHub Copilot + human curiosity.
Tested in transformer models, vision backbones, and micro-batch settings.

# AdamW-mini-SF

**AdamW に基づいた軽量かつスケジューリング不要な最適化手法 — 自動学習率調整＆AMPサポート対応。**

このオプティマイザは、[Adam-mini](https://github.com/zyushun/Adam-mini) を拡張し、以下の特徴を持ちます：

- 🚀 **省メモリな状態管理**：モーメント（`m`, `v`）を低精度（`float16` や `bfloat16`）で保持
- 🧠 **Schedule-Free な学習率調整**：スムーズな勾配ノルムを追跡し、`lr` を動的に調整（スケジューラー不要）
- 🛡️ **分離されたWeight Decay（AdamW形式）**：勾配とは独立した正則化処理
- ⚙️ **AMP / mixed precision に対応**：パラメータの dtype を自動検出し、`torch.amp` とシームレスに連携可能

## インストール

`adamw_mini_sf.py` をプロジェクトにコピーしてください。

## 使い方

```python
from adamw_mini_sf import AdamWminiScheduleFree
optimizer = AdamWminiScheduleFree(model.parameters(), lr=1e-3)
```

ライセンス
Apache License 2.0 — 詳細は LICENSE をご覧ください。

🤖 GitHub Copilot と人間の好奇心のコラボで誕生しました。
Transformer系モデルやマイクロバッチ学習などで実験・活用されています。
