"""Variable-aware SwinV2 super-resolution model for climate downscaling."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ClimateSwinConfig:
    variable_names: tuple[str, ...] = ("tmin", "tmax", "prcp")
    embed_dim: int = 96
    num_groups: int = 4
    blocks_per_group: int = 4
    num_heads: int = 6
    window_size: int = 8
    mlp_ratio: float = 2.0
    scale_factor: int = 4
    static_lr_channels: int = 4
    static_hr_channels: int = 4
    variable_dropout: float = 0.1
    drop_path: float = 0.1

    def __post_init__(self) -> None:
        if not self.variable_names:
            raise ValueError("At least one variable is required")
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.scale_factor < 1 or self.scale_factor & (self.scale_factor - 1):
            raise ValueError("scale_factor must be a positive power of two")
        if not 0.0 <= self.variable_dropout < 1.0:
            raise ValueError("variable_dropout must be in [0, 1)")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["variable_names"] = list(self.variable_names)
        return result

    @classmethod
    def from_dict(cls, raw: dict) -> "ClimateSwinConfig":
        values = dict(raw)
        values["variable_names"] = tuple(values["variable_names"])
        return cls(**values)


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * random.floor() / keep


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    batch, height, width, channels = x.shape
    x = x.view(
        batch,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, channels)


def window_reverse(windows: torch.Tensor, window_size: int, height: int, width: int) -> torch.Tensor:
    windows_per_image = (height // window_size) * (width // window_size)
    batch = windows.shape[0] // windows_per_image
    x = windows.view(
        batch,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)


class WindowAttentionV2(nn.Module):
    """SwinV2 scaled-cosine attention with continuous relative position bias."""

    def __init__(self, dim: int, window_size: int, num_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.logit_scale = nn.Parameter(torch.log(torch.ones(num_heads, 1, 1) * 10.0))
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.projection = nn.Linear(dim, dim)

        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )
        coordinates = torch.arange(-(window_size - 1), window_size, dtype=torch.float32)
        table = torch.stack(torch.meshgrid(coordinates, coordinates, indexing="ij"), dim=-1).unsqueeze(0)
        table = table / max(window_size - 1, 1) * 8.0
        table = torch.sign(table) * torch.log2(torch.abs(table) + 1.0) / math.log2(8.0)
        self.register_buffer("relative_coordinates_table", table, persistent=False)

        positions = torch.stack(
            torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij")
        ).flatten(1)
        relative = positions[:, :, None] - positions[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative.sum(-1), persistent=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_windows, tokens, channels = x.shape
        qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(batch_windows, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        scale = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
        attention = (query @ key.transpose(-2, -1)) * scale

        bias_table = 16.0 * torch.sigmoid(self.cpb_mlp(self.relative_coordinates_table)).view(-1, self.num_heads)
        relative_bias = bias_table[self.relative_position_index.view(-1)]
        relative_bias = relative_bias.view(tokens, tokens, self.num_heads).permute(2, 0, 1)
        attention = attention + relative_bias.unsqueeze(0).to(attention.dtype)
        if mask is not None:
            n_windows = mask.shape[0]
            attention = attention.view(batch_windows // n_windows, n_windows, self.num_heads, tokens, tokens)
            attention = attention + mask[None, :, None].to(attention.dtype)
            attention = attention.view(-1, self.num_heads, tokens, tokens)
        attention = attention.softmax(dim=-1)
        output = (attention @ value).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.projection(output)


class SwinV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.attention = WindowAttentionV2(dim, window_size, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.drop_path = DropPath(drop_path)

    def _attention_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros((1, height, width, 1), device=device)
        slices_h = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        slices_w = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        label = 0
        for row in slices_h:
            for col in slices_w:
                mask[:, row, col, :] = label
                label += 1
        windows = window_partition(mask, self.window_size).squeeze(-1)
        difference = windows.unsqueeze(1) - windows.unsqueeze(2)
        return difference.masked_fill(difference != 0, -100.0).masked_fill(difference == 0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        residual = x
        x = x.permute(0, 2, 3, 1)
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        padded_h, padded_w = x.shape[1:3]
        use_shift = self.shift_size if min(padded_h, padded_w) > self.window_size else 0
        shifted = torch.roll(x, shifts=(-use_shift, -use_shift), dims=(1, 2)) if use_shift else x
        windows = window_partition(shifted, self.window_size)
        mask = self._attention_mask(padded_h, padded_w, x.device) if use_shift else None
        attended = self.attention(windows, mask)
        shifted = window_reverse(attended, self.window_size, padded_h, padded_w)
        x = torch.roll(shifted, shifts=(use_shift, use_shift), dims=(1, 2)) if use_shift else shifted
        x = x[:, :height, :width]
        x = self.norm1(x).permute(0, 3, 1, 2)
        x = residual + self.drop_path(x)
        mlp = self.mlp(x.permute(0, 2, 3, 1))
        mlp = self.norm2(mlp).permute(0, 3, 1, 2)
        return x + self.drop_path(mlp)


class ResidualSwinGroup(nn.Module):
    def __init__(self, config: ClimateSwinConfig, drop_paths: Sequence[float]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinV2Block(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                window_size=config.window_size,
                shift_size=0 if index % 2 == 0 else config.window_size // 2,
                mlp_ratio=config.mlp_ratio,
                drop_path=drop_paths[index],
            )
            for index in range(config.blocks_per_group)
        ])
        self.convolution = nn.Conv2d(config.embed_dim, config.embed_dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for block in self.blocks:
            x = block(x)
        return residual + self.convolution(x)


class VariableAwareStem(nn.Module):
    def __init__(self, variable_count: int, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.variable_count = variable_count
        self.dropout = float(dropout)
        self.stems = nn.ModuleList([nn.Conv2d(1, dim, 3, padding=1) for _ in range(variable_count)])
        self.variable_embedding = nn.Parameter(torch.zeros(variable_count, dim))
        self.presence_embedding = nn.Embedding(2, dim)
        cross_heads = max(1, min(num_heads, 4))
        while dim % cross_heads:
            cross_heads -= 1
        self.cross_variable_attention = nn.MultiheadAttention(dim, cross_heads, batch_first=True)
        nn.init.trunc_normal_(self.variable_embedding, std=0.02)

    def _presence(self, batch: int, device: torch.device, variable_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if variable_mask is None:
            present = torch.ones(batch, self.variable_count, dtype=torch.bool, device=device)
        else:
            if variable_mask.shape != (batch, self.variable_count):
                raise ValueError(f"variable_mask must have shape {(batch, self.variable_count)}")
            present = variable_mask.to(device=device, dtype=torch.bool)
        if self.training and self.dropout > 0.0:
            present = present & (torch.rand(batch, self.variable_count, device=device) >= self.dropout)
        missing_all = ~present.any(dim=1)
        if missing_all.any():
            present[missing_all, 0] = True
        return present

    def forward(self, x: torch.Tensor, variable_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.shape[1] != self.variable_count:
            raise ValueError(f"Expected {self.variable_count} variables, got {x.shape[1]}")
        batch, _, height, width = x.shape
        present = self._presence(batch, x.device, variable_mask)
        features = torch.stack([stem(x[:, index:index + 1]) for index, stem in enumerate(self.stems)], dim=1)
        features = features.permute(0, 3, 4, 1, 2)  # B,H,W,V,C
        embeddings = self.variable_embedding[None, None, None]
        presence = self.presence_embedding(present.long())[:, None, None]
        tokens = (features + embeddings + presence).reshape(batch * height * width, self.variable_count, -1)
        key_padding = (~present)[:, None, :].expand(batch, height * width, self.variable_count)
        key_padding = key_padding.reshape(batch * height * width, self.variable_count)
        tokens, _ = self.cross_variable_attention(tokens, tokens, tokens, key_padding_mask=key_padding, need_weights=False)
        weights = present[:, None, None, :, None].to(tokens.dtype)
        tokens = tokens.view(batch, height, width, self.variable_count, -1)
        fused = (tokens * weights).sum(dim=3) / weights.sum(dim=3).clamp_min(1.0)
        return fused.permute(0, 3, 1, 2).contiguous()


class UpsampleFusionStage(nn.Module):
    def __init__(self, dim: int, terrain_channels: int) -> None:
        super().__init__()
        self.expand = nn.Conv2d(dim, dim * 4, 3, padding=1)
        self.fusion = nn.Sequential(
            nn.Conv2d(dim + terrain_channels, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, terrain: torch.Tensor) -> torch.Tensor:
        x = F.pixel_shuffle(self.expand(x), 2)
        terrain = F.interpolate(terrain, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x + self.fusion(torch.cat([x, terrain], dim=1))


class VariableDecoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 2, 3, padding=1),
            nn.GELU(),
        )
        self.output = nn.Conv2d(dim // 2, 1, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.body(x))


class ClimateSwin(nn.Module):
    """Shared variable-aware SwinV2 encoder with specialized reconstruction heads."""

    def __init__(self, config: ClimateSwinConfig) -> None:
        super().__init__()
        self.config = config
        variable_count = len(config.variable_names)
        self.variable_stem = VariableAwareStem(
            variable_count, config.embed_dim, config.num_heads, config.variable_dropout
        )
        self.static_lr_stem = nn.Conv2d(config.static_lr_channels, config.embed_dim, 3, padding=1)
        self.season_embedding = nn.Sequential(
            nn.Linear(2, config.embed_dim), nn.GELU(), nn.Linear(config.embed_dim, config.embed_dim)
        )
        self.shallow = nn.Conv2d(config.embed_dim, config.embed_dim, 3, padding=1)

        total_blocks = config.num_groups * config.blocks_per_group
        drop_paths = torch.linspace(0.0, config.drop_path, total_blocks).tolist()
        self.groups = nn.ModuleList()
        for group_index in range(config.num_groups):
            start = group_index * config.blocks_per_group
            self.groups.append(ResidualSwinGroup(config, drop_paths[start:start + config.blocks_per_group]))
        self.encoder_output = nn.Conv2d(config.embed_dim, config.embed_dim, 3, padding=1)

        terrain_channels = config.static_hr_channels + 1  # HR predictors plus elevation anomaly
        stages = int(math.log2(config.scale_factor))
        self.upsampling = nn.ModuleList([
            UpsampleFusionStage(config.embed_dim, terrain_channels) for _ in range(stages)
        ])
        self.decoders = nn.ModuleList([VariableDecoder(config.embed_dim) for _ in config.variable_names])

    def _terrain_features(self, static_lr: torch.Tensor, static_hr: torch.Tensor) -> torch.Tensor:
        lr_elevation = F.interpolate(
            static_lr[:, :1], size=static_hr.shape[-2:], mode="bilinear", align_corners=False
        )
        anomaly = static_hr[:, :1] - lr_elevation
        return torch.cat([static_hr, anomaly], dim=1)

    def forward(
        self,
        dynamic_lr: torch.Tensor,
        static_lr: torch.Tensor,
        static_hr: torch.Tensor,
        season: torch.Tensor,
        variable_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        expected_hr = tuple(size * self.config.scale_factor for size in dynamic_lr.shape[-2:])
        if tuple(static_hr.shape[-2:]) != expected_hr:
            raise ValueError(f"static_hr has shape {static_hr.shape[-2:]}, expected {expected_hr}")
        if static_lr.shape[-2:] != dynamic_lr.shape[-2:]:
            raise ValueError("static_lr and dynamic_lr must share a spatial grid")
        x = self.variable_stem(dynamic_lr, variable_mask)
        x = x + self.static_lr_stem(static_lr)
        x = x + self.season_embedding(season)[:, :, None, None]
        shallow = self.shallow(x)
        x = shallow
        for group in self.groups:
            x = group(x)
        x = shallow + self.encoder_output(x)

        terrain = self._terrain_features(static_lr, static_hr)
        for stage in self.upsampling:
            x = stage(x, terrain)
        residuals = torch.cat([decoder(x) for decoder in self.decoders], dim=1)
        baseline = F.interpolate(dynamic_lr, scale_factor=self.config.scale_factor, mode="bilinear", align_corners=False)
        return baseline + residuals


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
