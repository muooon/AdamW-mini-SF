import torch
from torch.optim.optimizer import Optimizer

class AdamWminiScheduleFree(Optimizer):
    """
    AdamW-mini + ScheduleFree: 省メモリ・高速化 + スケジューリング不要の自動学習率調整
    - m/vはfloat16で保持
    - Weight DecayはAdamW方式
    - 学習率は損失や勾配ノルムに応じて自動調整（ScheduleFree）
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, min_lr=1e-6, max_lr=1e-2, schedulefree_alpha=0.05):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        min_lr=min_lr, max_lr=max_lr, schedulefree_alpha=schedulefree_alpha)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # mixed precision対応: grad, m, vはパラメータのdtypeに合わせる
                param_dtype = p.data.dtype
                grad = p.grad.data.to(param_dtype)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # bfloat16/float16/float32どれでもOK
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=param_dtype)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=param_dtype)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                # m, vをパラメータのdtypeで保持（bfloat16/float16/float32/amp対応）
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # bias correction省略
                # ScheduleFree: 自動学習率調整
                grad_norm = grad.norm().item()
                if 'grad_norm_ema' not in state:
                    state['grad_norm_ema'] = grad_norm
                else:
                    state['grad_norm_ema'] = (1 - group['schedulefree_alpha']) * state['grad_norm_ema'] + group['schedulefree_alpha'] * grad_norm
                # 学習率を勾配ノルムに応じて自動調整
                auto_lr = max(min(group['max_lr'], group['lr'] * (state['grad_norm_ema'] / (grad_norm + 1e-12))), group['min_lr'])
                # 完全なスケーリング: step_sizeにauto_lrを直接かける
                step_size = auto_lr
                denom = (exp_avg_sq.to(torch.float32).sqrt() + group['eps']).to(param_dtype)
                # AdamW weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * auto_lr)
                # パラメータ更新
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        return loss
