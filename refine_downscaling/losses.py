"""Balanced multivariable objectives and physical consistency penalties."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transforms import TransformSpec, inverse_torch


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dx_pred = prediction[..., :, 1:] - prediction[..., :, :-1]
    dx_true = target[..., :, 1:] - target[..., :, :-1]
    dy_pred = prediction[..., 1:, :] - prediction[..., :-1, :]
    dy_true = target[..., 1:, :] - target[..., :-1, :]
    mask_x = mask[..., :, 1:] * mask[..., :, :-1]
    mask_y = mask[..., 1:, :] * mask[..., :-1, :]
    return masked_mean(torch.abs(dx_pred - dx_true), mask_x) + masked_mean(
        torch.abs(dy_pred - dy_true), mask_y
    )


class MultivariableLoss(nn.Module):
    def __init__(
        self,
        variable_names: Sequence[str],
        specs: Mapping[str, TransformSpec],
        variable_weights: Mapping[str, float] | None = None,
        huber_weight: float = 1.0,
        mae_weight: float = 0.1,
        gradient_weight: float = 0.1,
        temperature_order_weight: float = 0.05,
        precipitation_conservation_weight: float = 0.05,
        scale_factor: int = 4,
    ) -> None:
        super().__init__()
        self.variable_names = tuple(variable_names)
        self.specs = dict(specs)
        self.variable_weights = {
            name: float((variable_weights or {}).get(name, 1.0)) for name in self.variable_names
        }
        self.huber_weight = float(huber_weight)
        self.mae_weight = float(mae_weight)
        self.gradient_weight = float(gradient_weight)
        self.temperature_order_weight = float(temperature_order_weight)
        self.precipitation_conservation_weight = float(precipitation_conservation_weight)
        self.scale_factor = int(scale_factor)

    def _inverse(self, values: torch.Tensor, channel: int) -> torch.Tensor:
        return inverse_torch(values[:, channel:channel + 1], self.specs[self.variable_names[channel]])

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        dynamic_lr: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction/target shape mismatch: {prediction.shape}, {target.shape}")
        components: Dict[str, torch.Tensor] = {}
        total = prediction.new_zeros(())
        for channel, name in enumerate(self.variable_names):
            pred = prediction[:, channel:channel + 1]
            truth = target[:, channel:channel + 1]
            huber = masked_mean(F.smooth_l1_loss(pred, truth, reduction="none"), loss_mask)
            mae = masked_mean(torch.abs(pred - truth), loss_mask)
            gradients = gradient_loss(pred, truth, loss_mask)
            channel_loss = (
                self.huber_weight * huber
                + self.mae_weight * mae
                + self.gradient_weight * gradients
            ) * self.variable_weights[name]
            components[f"{name}_data"] = channel_loss
            total = total + channel_loss

        if "tmin" in self.variable_names and "tmax" in self.variable_names:
            tmin_index = self.variable_names.index("tmin")
            tmax_index = self.variable_names.index("tmax")
            tmin = self._inverse(prediction, tmin_index)
            tmax = self._inverse(prediction, tmax_index)
            order = masked_mean(F.relu(tmin - tmax), loss_mask)
            components["temperature_order"] = order
            total = total + self.temperature_order_weight * order

        precip_names = {"pr", "prcp", "precip", "precipitation"}
        precip_index = next(
            (index for index, name in enumerate(self.variable_names) if name in precip_names), None
        )
        if precip_index is not None and self.precipitation_conservation_weight > 0.0:
            prediction_raw = self._inverse(prediction, precip_index).clamp_min(0.0)
            lr_raw = self._inverse(dynamic_lr, precip_index).clamp_min(0.0)
            coarse_prediction = F.avg_pool2d(
                prediction_raw, kernel_size=self.scale_factor, stride=self.scale_factor
            )
            coarse_mask = F.avg_pool2d(
                loss_mask, kernel_size=self.scale_factor, stride=self.scale_factor
            )
            conservation = masked_mean(
                torch.abs(torch.log1p(coarse_prediction) - torch.log1p(lr_raw)),
                (coarse_mask > 0.999).to(coarse_mask.dtype),
            )
            components["precipitation_conservation"] = conservation
            total = total + self.precipitation_conservation_weight * conservation

        components["total"] = total
        return total, components
