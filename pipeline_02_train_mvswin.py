#!/usr/bin/env python3
"""Train the patch-based multivariable ClimateSwin downscaler."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from climate_downscaling.data import MultivariablePatchDataset, load_manifest
from climate_downscaling.distributed import (
    all_reduce_mean,
    barrier,
    cleanup_distributed,
    initialize_distributed,
)
from climate_downscaling.losses import MultivariableLoss
from climate_downscaling.model import ClimateSwin, ClimateSwinConfig, count_parameters
from climate_downscaling.transforms import specs_from_manifest


def load_training_layout(data_dir: Path):
    raw = json.loads((Path(data_dir) / "manifest.json").read_text())
    layout = raw.get("storage_layout")
    if layout == "variable_separable_npy":
        return load_manifest(data_dir), MultivariablePatchDataset
    if layout == "netcdf_patch_index":
        from climate_downscaling.stage2_data import (
            Stage2NetCDFPatchDataset,
            load_stage2_manifest,
        )

        return load_stage2_manifest(data_dir), Stage2NetCDFPatchDataset
    raise ValueError(f"Unsupported training storage layout: {layout!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data/daymet_mv_1980_1990"))
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/runs/climateswin_v1"))
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume", type=Path)
    checkpoint_group.add_argument(
        "--init-backbone", type=Path,
        help="Warm-start compatible encoder/fusion weights; optimizer and upsampling head start fresh",
    )
    parser.add_argument(
        "--variables", nargs="+",
        help="Prepared variables to train, in channel order (default: every manifest variable)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4, help="Per process")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patches-per-day", type=int, default=32)
    parser.add_argument("--validation-patches-per-day", type=int, default=8)
    parser.add_argument("--core-size", type=int, default=16)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--num-groups", type=int, default=4)
    parser.add_argument("--blocks-per-group", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--variable-dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--mae-weight", type=float, default=0.1)
    parser.add_argument("--gradient-weight", type=float, default=0.1)
    parser.add_argument("--temperature-order-weight", type=float, default=0.05)
    parser.add_argument("--precipitation-conservation-weight", type=float, default=0.05)
    parser.add_argument("--amp", action="store_true", help="Use bfloat16 autocast on GPU")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size: int, workers: int, distributed: bool, shuffle: bool):
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=shuffle)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=False,
        persistent_workers=workers > 0,
        drop_last=shuffle,
    )
    return loader, sampler


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def learning_rate_multiplier(epoch: int, warmup: int, total: int) -> float:
    if warmup > 0 and epoch < warmup:
        return float(epoch + 1) / warmup
    progress = (epoch - warmup) / max(total - warmup - 1, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def initialize_backbone(model: torch.nn.Module, checkpoint_path: Path) -> dict:
    """Load matching Stage-1 representation weights, excluding scale-specific heads."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_config = ClimateSwinConfig.from_dict(checkpoint["model_config"])
    target = unwrap_model(model)
    if source_config.variable_names != target.config.variable_names:
        raise ValueError(
            "Backbone checkpoint variables do not match: "
            f"{source_config.variable_names} != {target.config.variable_names}"
        )
    prefixes = (
        "variable_stem.",
        "static_lr_stem.",
        "season_embedding.",
        "shallow.",
        "groups.",
        "encoder_output.",
    )
    target_state = target.state_dict()
    transferred = {
        name: value
        for name, value in checkpoint["model"].items()
        if name.startswith(prefixes)
        and name in target_state
        and target_state[name].shape == value.shape
    }
    if not transferred:
        raise ValueError(f"No compatible backbone weights found in {checkpoint_path}")
    target.load_state_dict(transferred, strict=False)
    transferable = [name for name in target_state if name.startswith(prefixes)]
    return {
        "checkpoint": str(checkpoint_path),
        "source_scale_factor": source_config.scale_factor,
        "loaded_tensors": len(transferred),
        "eligible_tensors": len(transferable),
        "skipped_tensors": len(transferable) - len(transferred),
    }


def recover_early_stop_patience(history_path: Path, checkpoint: dict) -> int:
    if "early_stop_patience_count" in checkpoint:
        return int(checkpoint["early_stop_patience_count"])
    if not history_path.exists():
        return 0
    best_validation = float(checkpoint["best_validation"])
    checkpoint_epoch = int(checkpoint["epoch"])
    patience = 0
    records = [
        json.loads(line) for line in history_path.read_text().splitlines()
        if line.strip()
    ]
    for record in reversed(records):
        if int(record["epoch"]) > checkpoint_epoch:
            continue
        if float(record["validation"]["total"]) <= best_validation + 1e-12:
            break
        patience += 1
    return patience


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_validation: float,
    early_stop_patience_count: int,
    model_config: ClimateSwinConfig,
    data_manifest: dict,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "best_validation": best_validation,
            "early_stop_patience_count": early_stop_patience_count,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model_config": model_config.to_dict(),
            "variables": list(model_config.variable_names),
            "data_manifest": data_manifest,
        },
        temporary,
    )
    os.replace(temporary, path)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    objective: MultivariableLoss,
    device: torch.device,
    amp: bool,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip: float = 1.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums: dict[str, torch.Tensor] = {}
    steps = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            batch = move_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp):
                prediction = model(
                    batch["lr"], batch["static_lr"], batch["static_hr"], batch["season"]
                )
                loss, components = objective(
                    prediction, batch["target"], batch["lr"], batch["loss_mask"]
                )
            if training:
                loss.backward()
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
            for name, value in components.items():
                sums[name] = sums.get(name, torch.zeros((), device=device)) + value.detach().float()
            steps += 1
    if steps == 0:
        raise RuntimeError("Data loader yielded no batches; reduce batch size or increase samples")
    return {name: float(all_reduce_mean(value / steps).cpu()) for name, value in sums.items()}


def main() -> None:
    args = build_parser().parse_args()
    context = initialize_distributed()
    try:
        seed_everything(args.seed + context.rank)
        manifest, patch_dataset_class = load_training_layout(args.data_dir)
        variables = tuple(args.variables or manifest["variables"])
        config = ClimateSwinConfig(
            variable_names=variables,
            embed_dim=args.embed_dim,
            num_groups=args.num_groups,
            blocks_per_group=args.blocks_per_group,
            num_heads=args.num_heads,
            window_size=args.window_size,
            mlp_ratio=args.mlp_ratio,
            scale_factor=int(manifest["scale_factor"]),
            variable_dropout=args.variable_dropout,
            drop_path=args.drop_path,
        )
        train_dataset = patch_dataset_class(
            args.data_dir,
            split="train",
            core_size=args.core_size,
            halo=args.halo,
            patches_per_day=args.patches_per_day,
            random_patches=True,
            variable_names=variables,
        )
        validation_dataset = patch_dataset_class(
            args.data_dir,
            split="val",
            core_size=args.core_size,
            halo=args.halo,
            patches_per_day=args.validation_patches_per_day,
            random_patches=False,
            variable_names=variables,
        )
        distributed = context.world_size > 1
        train_loader, train_sampler = make_loader(
            train_dataset, args.batch_size, args.num_workers, distributed, shuffle=True
        )
        validation_loader, _ = make_loader(
            validation_dataset, args.batch_size, args.num_workers, distributed, shuffle=False
        )

        model = ClimateSwin(config).to(context.device)
        backbone_initialization = None
        if args.init_backbone:
            backbone_initialization = initialize_backbone(model, args.init_backbone)
            if context.is_main:
                print(
                    "Initialized Stage-2 representation from "
                    f"{args.init_backbone}: {backbone_initialization['loaded_tensors']}/"
                    f"{backbone_initialization['eligible_tensors']} tensors",
                    flush=True,
                )
        if distributed:
            device_ids = [context.device.index] if context.device.type == "cuda" else None
            model = DistributedDataParallel(model, device_ids=device_ids)
        objective = MultivariableLoss(
            variable_names=variables,
            specs=specs_from_manifest(manifest),
            mae_weight=args.mae_weight,
            gradient_weight=args.gradient_weight,
            temperature_order_weight=args.temperature_order_weight,
            precipitation_conservation_weight=args.precipitation_conservation_weight,
            scale_factor=config.scale_factor,
        ).to(context.device)
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: learning_rate_multiplier(epoch, args.warmup_epochs, args.epochs),
        )

        history_path = args.run_dir / "history.jsonl"
        start_epoch = 0
        best_validation = float("inf")
        patience = 0
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            if checkpoint["model_config"] != config.to_dict():
                raise ValueError("Resume checkpoint model configuration does not match CLI configuration")
            unwrap_model(model).load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_validation = float(checkpoint["best_validation"])
            patience = recover_early_stop_patience(history_path, checkpoint)
            if context.is_main:
                print(
                    f"Resuming after epoch {start_epoch - 1}; "
                    f"best_validation={best_validation:.8f}; patience={patience}",
                    flush=True,
                )

        if context.is_main:
            args.run_dir.mkdir(parents=True, exist_ok=True)
            run_config = vars(args).copy()
            run_config.update({
                "data_dir": str(args.data_dir),
                "run_dir": str(args.run_dir),
                "resume": str(args.resume) if args.resume else None,
                "init_backbone": str(args.init_backbone) if args.init_backbone else None,
                "backbone_initialization": backbone_initialization,
                "model": config.to_dict(),
                "parameters": count_parameters(unwrap_model(model)),
                "world_size": context.world_size,
            })
            (args.run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
            print(json.dumps(run_config, indent=2), flush=True)
        barrier()

        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_metrics = run_epoch(
                model, train_loader, objective, context.device, args.amp,
                optimizer=optimizer, gradient_clip=args.gradient_clip,
            )
            validation_metrics = run_epoch(
                model, validation_loader, objective, context.device, args.amp,
            )
            scheduler.step()
            validation_total = validation_metrics["total"]
            improved = validation_total < best_validation
            if improved:
                best_validation = validation_total
                patience = 0
            else:
                patience += 1

            if context.is_main:
                record = {
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "best_validation": best_validation,
                }
                with history_path.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                save_checkpoint(
                    args.run_dir / "last.pt", model, optimizer, scheduler, epoch,
                    best_validation, patience, config, manifest,
                )
                if improved:
                    save_checkpoint(
                        args.run_dir / "best.pt", model, optimizer, scheduler, epoch,
                        best_validation, patience, config, manifest,
                    )
                print(json.dumps(record), flush=True)
            stop = patience >= args.early_stop_patience
            if distributed:
                stop_tensor = torch.tensor(int(stop), device=context.device)
                dist.broadcast(stop_tensor, src=0)
                stop = bool(stop_tensor.item())
            if stop:
                if context.is_main:
                    print(f"Early stopping after epoch {epoch}", flush=True)
                break
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
