#!/usr/bin/env python3
"""
Warm-up Learning Rate Scheduler
Постепенно увеличивает learning rate в начале обучения
"""

import torch
import math
from typing import Optional


class WarmupScheduler:
    """Простой warm-up scheduler для learning rate"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 3,
        base_lr: Optional[float] = None,
        warmup_start_factor: float = 0.1,
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Количество эпох для warm-up
            base_lr: Базовый learning rate (по умолчанию берется из optimizer)
            warmup_start_factor: Начальный коэффициент (0.1 = начинаем с 10% от base_lr)
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_start_factor = warmup_start_factor

        # Получаем базовый learning rate
        if base_lr is None:
            self.base_lr = optimizer.param_groups[0]["lr"]
        else:
            self.base_lr = base_lr

        self.current_epoch = 0

        # Сохраняем исходные learning rates для всех param_groups
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch: Optional[int] = None):
        """
        Обновляет learning rate на новой эпохе

        Args:
            epoch: Номер эпохи (если None, использует внутренний счетчик)
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        # Вычисляем коэффициент для текущей эпохи
        if self.current_epoch <= self.warmup_epochs:
            # Warm-up фаза: линейное увеличение от start_factor до 1.0
            warmup_factor = self.warmup_start_factor + (
                1.0 - self.warmup_start_factor
            ) * (self.current_epoch / self.warmup_epochs)
        else:
            # После warm-up: используем полный learning rate
            warmup_factor = 1.0

        # Применяем коэффициент ко всем param_groups
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.base_lrs[i] * warmup_factor

        return warmup_factor

    def get_current_lr(self) -> float:
        """Возвращает текущий learning rate"""
        return self.optimizer.param_groups[0]["lr"]

    def get_warmup_info(self) -> dict:
        """Возвращает информацию о warm-up"""
        current_lr = self.get_current_lr()
        warmup_factor = current_lr / self.base_lr if self.base_lr > 0 else 0

        return {
            "epoch": self.current_epoch,
            "warmup_epochs": self.warmup_epochs,
            "base_lr": self.base_lr,
            "current_lr": current_lr,
            "warmup_factor": warmup_factor,
            "is_warmup_phase": self.current_epoch <= self.warmup_epochs,
        }


def create_warmup_scheduler(
    optimizer: torch.optim.Optimizer, is_resume: bool = False, warmup_epochs: int = 3
) -> Optional[WarmupScheduler]:
    """
    Создает warm-up scheduler если нужно

    Args:
        optimizer: PyTorch optimizer
        is_resume: Если True, создает warm-up scheduler
        warmup_epochs: Количество эпох для warm-up

    Returns:
        WarmupScheduler или None
    """
    if is_resume:
        return WarmupScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            warmup_start_factor=0.2,  # Начинаем с 20% от базового LR
        )
    else:
        return None


def test_warmup_scheduler():
    """Тест warm-up scheduler"""
    print("🧪 Testing WarmupScheduler")

    # Создаем dummy optimizer
    import torch.nn as nn

    model = nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Создаем scheduler
    scheduler = WarmupScheduler(optimizer, warmup_epochs=5, warmup_start_factor=0.1)

    print(f"\nBase LR: {scheduler.base_lr:.6f}")
    print(f"Warmup epochs: {scheduler.warmup_epochs}")
    print(f"Start factor: {scheduler.warmup_start_factor}")
    print("\nWarm-up progression:")

    for epoch in range(1, 8):
        factor = scheduler.step(epoch)
        info = scheduler.get_warmup_info()

        print(
            f"Epoch {epoch:2d}: LR={info['current_lr']:.6f} (factor={factor:.3f}) {'[WARMUP]' if info['is_warmup_phase'] else '[NORMAL]'}"
        )


if __name__ == "__main__":
    test_warmup_scheduler()
