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
If dtype is omitted, the optimizer will follow p.data.dtype to determine the internal precision. However, to enable half precision (for memory savings), it must be explicitly specified:
```
optimizer = AdamWminiScheduleFree(model.parameters(), lr=1e-3, dtype=torch.float16)
```
With this, optimizer states like exp_avg and exp_avg_sq will be stored in half precision, allowing for both memory and performance optimizations.

License
Apache License 2.0 — see LICENSE for details.

Built with 🤖 GitHub Copilot + human curiosity.
Tested in transformer models, vision backbones, and micro-batch settings.

## Acknowledgments

This project builds upon the excellent work of [Adam-mini](https://github.com/zyushun/Adam-mini) by @zyushun — thank you for your contributions to lightweight optimizers.

Thanks also to the open-source community behind PyTorch, and to GitHub Copilot for being an inspiring coding partner.

We are grateful to the research community whose ideas around AdamW, Schedule-Free optimization, and mixed precision have made this possible.



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
dtype を省略すれば p.data.dtype に従って状態が作られますが、半精度（省メモリ化）を有効にしたい場合は明示が必要です：
```
optimizer = AdamWminiScheduleFree(model.parameters(), lr=1e-3, dtype=torch.float16)
```
これで exp_avg / exp_avg_sq などのモーメントが半精度で保持され、メモリ・速度の最適化が効きます。

ライセンス
Apache License 2.0 — 詳細は LICENSE をご覧ください。

🤖 GitHub Copilot と人間の好奇心のコラボで誕生しました。
Transformer系モデルやマイクロバッチ学習などで実験・活用されています。

## 謝辞（Acknowledgments）

本プロジェクトは、[@zyushun](https://github.com/zyushun) 氏による [Adam-mini](https://github.com/zyushun/Adam-mini) の素晴らしい先行研究と実装に多くを学び、その上に構築しています。軽量かつ高性能な最適化器の礎を築いていただき、深く感謝申し上げます。

また、PyTorch および OSS コミュニティの皆さま、Schedule-Free 最適化や mixed precision 学習に関する研究を築いてきた研究者の方々の知見に、心より敬意を表します。

さらに、本実装にあたっては GitHub Copilot との協働も大きな助けとなりました。AI支援による開発の可能性に感謝するとともに、これからも人間とAIの共創が広がることを願っています。

