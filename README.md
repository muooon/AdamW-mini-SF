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
```python
optimizer = AdamWminiScheduleFree(model.parameters(), lr=1e-3, dtype=torch.float16)
```
With this, optimizer states like exp_avg and exp_avg_sq will be stored in half precision, allowing for both memory and performance optimizations.

License
Apache License 2.0 — see LICENSE for details.

Built with 🤖 GitHub Copilot + human curiosity.

## Acknowledgments

This project builds upon the excellent work of [Adam-mini](https://github.com/zyushun/Adam-mini) by @zyushun — thank you for your contributions to lightweight optimizers.

Thanks also to the open-source community behind PyTorch, and to GitHub Copilot for being an inspiring coding partner.

We are grateful to the research community whose ideas around AdamW, Schedule-Free optimization, and mixed precision have made this possible.
 
![AdamW-mini-ScheduleFree00](https://github.com/muooon/adamw-mini-ScheduleFree/blob/main/AdamW-mini-ScheduleFree00.png?raw=true)
The test code is provided at the end.
 
# AdamW-mini-SF

**AdamW に基づいた軽量かつスケジューリング不要な最適化手法 — 自動学習率調整＆AMPサポート対応。**

このオプティマイザは、[Adam-mini](https://github.com/zyushun/Adam-mini) を拡張し、以下の特徴を持ちます：

- 🚀 **省メモリな状態管理**：モーメント(`m`, `v`)を低精度(`float16` や `bfloat16`)で保持
- 🧠 **Schedule-Free な学習率調整**：スムーズな勾配ノルムを追跡し、`lr` を動的に調整(スケジューラー不要)
- 🛡️ **分離されたWeight Decay(AdamW形式)**：勾配とは独立した正則化処理
- ⚙️ **AMP / mixed precision に対応**：パラメータの dtype を自動検出し、`torch.amp` とシームレスに連携可能

## インストール

`adamw_mini_sf.py` をプロジェクトにコピーしてください。

## 使い方

```python
from adamw_mini_sf import AdamWminiScheduleFree
optimizer = AdamWminiScheduleFree(model.parameters(), lr=1e-3)
```
dtype を省略すれば p.data.dtype に従って状態が作られますが、半精度（省メモリ化）を有効にしたい場合は明示が必要です：
```python
optimizer = AdamWminiScheduleFree(model.parameters(), lr=1e-3, dtype=torch.float16)
```
これで exp_avg / exp_avg_sq などのモーメントが半精度で保持され、メモリ・速度の最適化が効きます。

ライセンス
Apache License 2.0 — 詳細は LICENSE をご覧ください。

🤖 GitHub Copilot と人間の好奇心のコラボで誕生しました。

## 謝辞(Acknowledgments)

本プロジェクトは、[@zyushun](https://github.com/zyushun) 氏による [Adam-mini](https://github.com/zyushun/Adam-mini) の素晴らしい先行研究と実装に多くを学び、その上に構築しています。軽量かつ高性能な最適化器の礎を築いていただき、深く感謝申し上げます。

また、PyTorch および OSS コミュニティの皆さま、Schedule-Free 最適化や mixed precision 学習に関する研究を築いてきた研究者の方々の知見に、心より敬意を表します。

さらに、本実装にあたっては GitHub Copilot との協働も大きな助けとなりました。AI支援による開発の可能性に感謝するとともに、これからも人間とAIの共創が広がることを願っています。

## Benchmark Code (for Reproducibility)
### 比較実験コード（再現用）

Below is a test script that compares the processing speed and memory usage of AdamW and AdamW-mini-ScheduleFree. You can copy and run it as-is to reproduce the results.

以下は、AdamWとAdamW-mini-ScheduleFreeの処理速度・メモリ使用量を比較したテストコードです。再現性のため、そのまま貼り付けて実行できます。

<details>
<summary>Show Test Code | テストコードを表示</summary>

```python
import torch, time
import matplotlib.pyplot as plt
from torch import nn, utils
from torch.optim import AdamW
from torch.utils.checkpoint import checkpoint_sequential

from adamw_mini_sf import AdamWminiScheduleFree

import matplotlib
matplotlib.rcParams['font.family'] = 'Meiryo'  # Windowsの場合

# モデル定義（3ブロックに分けてcheckpointing対応）
class CheckpointedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048)
        )
    def forward(self, x):
        return checkpoint_sequential(self.seq, 3, x, use_reentrant=False)


# 初期化＆fp16化
model = CheckpointedModel().cuda() #.half()

# データもfp16
x = torch.randn(16, 2048, dtype=torch.float16, device='cuda', requires_grad=True)
y = torch.randn(16, 2048, dtype=torch.float16, device='cuda')
loss_fn = nn.MSELoss()

optimizers = {
    "AdamW": lambda: AdamW(model.parameters(), lr=1e-3),
    "AdamW-mini-SF": lambda: AdamWminiScheduleFree(model.parameters(), lr=1e-3, dtype=torch.float16)
}

records = {}
for name, opt_fn in optimizers.items():
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    model.apply(lambda m: hasattr(m, "reset_parameters") and m.reset_parameters())

    mem_log, time_log = [], []
    optimizer = opt_fn()
    scaler = torch.cuda.amp.GradScaler()  # AMPと併用可

    for _ in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        mem_mb = torch.cuda.memory_allocated() / 1024**2
        mem_log.append(mem_mb)
        time_log.append((t1 - t0) * 1000)

    records[name] = {"mem": mem_log, "time": time_log}

# グラフ描画
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for name, data in records.items():
    plt.plot(data["mem"], label=name)
plt.ylabel("VRAM使用量 (MB)")
plt.xlabel("Iteration")
plt.title("メモリ使用量の比較")
plt.legend()

plt.subplot(1, 2, 2)
for name, data in records.items():
    plt.plot(data["time"], label=name)
plt.ylabel("1ステップ時間 (ms)")
plt.xlabel("Iteration")
plt.title("処理時間の比較")
plt.legend()

plt.tight_layout()
plt.show()
```