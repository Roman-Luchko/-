import torch
import segmentation_models_pytorch as smp
import torch.nn as nn
import numpy as np


class DiceLoss(nn.Module):
    """Используем встроенный Dice Loss из SMP."""

    def __init__(self):
        super(DiceLoss, self).__init__()
        # Указываем mode='binary' для бинарной сегментации
        self.dice_loss = smp.losses.DiceLoss(mode='binary')

    def forward(self, input, target):
        """Расчет Dice Loss."""
        # Применяем сигмоиду на выходах модели перед расчетом Dice Loss
        input = torch.sigmoid(input)
        return self.dice_loss(input, target)


def dice_coefficient(prediction, target):
    """Используем встроенную метрику Dice из SMP."""

    # В SMP есть функция для вычисления Dice коэффициента
    return smp.metrics.dice_score(prediction, target)
