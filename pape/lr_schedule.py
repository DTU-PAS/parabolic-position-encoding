import math

import torch


class CosineDecayLRSchedule(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        final_lr: float,
        max_lr: float,
        total_steps: int,
        warmup_steps: int,
        last_epoch=-1,
    ):
        self.final_lr = final_lr
        self.max_lr = max_lr
        self.diff_lr = max_lr - final_lr
        self.decay_steps = total_steps - warmup_steps
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Decay the learning rate with half-cycle cosine after warmup

        Adapted from https://github.com/facebookresearch/mae/blob/main/util/lr_sched.py
        """
        step = self._step_count
        if step < self.warmup_steps:
            lr = self.max_lr * step / self.warmup_steps
        else:
            decayed_steps = step - self.warmup_steps
            lr = self.final_lr + self.diff_lr * 0.5 * (1.0 + math.cos(math.pi * decayed_steps / self.decay_steps))

        return [lr for _ in self.optimizer.param_groups]
