#!/usr/bin/env python
# Frontier-ready SRGAN training/eval (ROCm + DDP, one-rank-per-GPU)
from mpi4py import MPI
import os
import socket
import json
import pickle
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import matplotlib.pyplot as plt

# ---- Project-local modules (must be importable) ----
from srgan_torch import SRGAN_g_lr_26, SRGAN_d_lr_odd
from dataread import read_saved_data
from loss_torch import WithLoss_init, WithLoss_G, WithLoss_D

# ============================== CLI ARGS ===============================

def build_parser():
    p = argparse.ArgumentParser(description="Frontier ROCm SRGAN (DDP)")
    p.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    p.add_argument('--batch-size', type=int, default=128, help='Per-GPU batch size')
    p.add_argument('--amp', action='store_true', help='Enable bfloat16 autocast (MI250X)')
    p.add_argument('--version', type=str, default='dy_v0.1', help='Run/version tag for outputs')
    p.add_argument("--year-start", type=int, default=1980)
    p.add_argument("--year-end", type=int, default=2014)
    p.add_argument('--base-dir', type=str, default='/lustre/orion/proj-shared/cli138/7hn/SRGAN_3hr')
    p.add_argument('--initial-training', action='store_true', help='Run the pretrain/init phase')
    p.add_argument('--read-raw', action='store_true', help='Regenerate data via daymetread()')
    p.add_argument('--var', type=str, default='temp')

    # Epochs / schedulers (keep your defaults)
    p.add_argument('--n-epoch-init', type=int, default=50)
    p.add_argument('--n-epoch', type=int, default=100)

    # Loss weights
    p.add_argument('--w1-fn1', type=float, default=1e-4)
    p.add_argument('--w2-fn2', type=float, default=1e3)

    # Dataloader workers (defaults tuned for Frontier example)
    p.add_argument('--num-workers', type=int, default=7)

    p.add_argument("--master_addr", type=str, required=True)
    p.add_argument("--master_port", type=str, required=True)

    return p


# ====================== DISTRIBUTED / ROCm HELPERS =====================

def is_dist():
    # print("is_dist() check:", torch.distributed.is_available(), torch.distributed.is_initialized())
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank():
    return torch.distributed.get_rank() if is_dist() else 0

def get_world_size():
    return torch.distributed.get_world_size() if is_dist() else 1


# ============================== UTILITIES ==============================

def log_all_gpu_memory(device, local_rank):
    """Gather and print GPU memory stats from all ranks."""
    alloc = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2

    # Create per-rank dict
    stats = {
        "rank": dist.get_rank(),
        "gpu": local_rank,
        "alloc": round(alloc, 1),
        "reserved": round(reserved, 1)
    }

    # Gather stats from all ranks
    world_size = dist.get_world_size()
    all_stats = [None for _ in range(world_size)]
    dist.all_gather_object(all_stats, stats)

    # Print only on rank 0
    if dist.get_rank() == 0:
        print("=== GPU Memory Summary ===")
        for s in sorted(all_stats, key=lambda x: x["gpu"]):
            print(f"GPU {s['gpu']} (rank {s['rank']}): "
                  f"allocated={s['alloc']} MB, reserved={s['reserved']} MB")
        print("==========================\n")

def report_topology(args, device, local_rank):
    """Simplified topology report for Frontier ROCm DDP runs."""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    dev_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"
    current_dev = torch.cuda.current_device() if torch.cuda.is_available() else -1

    # CPU affinity
    try:
        cpu_affinity = sorted(os.sched_getaffinity(0))
        cpus_assigned = len(cpu_affinity)
    except AttributeError:
        cpus_assigned = int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        cpu_affinity = []

    host = socket.gethostname()

    visible = torch.cuda.device_count()
    physical = os.getenv("SLURM_GPUS_ON_NODE", "unknown")
    total = world * visible

    print(f"\n[Rank {rank}/{world} @ {host}]")
    print(f"  ├─ Device         : {device} (local_rank={local_rank}, {dev_name})")
    print(f"  ├─ Current GPU ID : {current_dev}")
    print(f"  ├─ Visible per rank  : {visible}")
    print(f"  ├─ Physical per node  : {physical}")
    print(f"  ├─ Total in job       : {total}", flush=True)
    print(f"  ├─ CPUs assigned  : {cpus_assigned}  | affinity={cpu_affinity}")
    print(f"  ├─ Batch size     : {getattr(args, 'batch_size', 'n/a')}  | num_workers={getattr(args, 'num_workers', 'n/a')}")
    print(f"  └─ Master address : {args.master_addr}:{args.master_port}\n", flush=True)




class TrainData(Dataset):
    def __init__(self, lr_data, hr_data):
        self.lr_data = lr_data
        self.hr_data = hr_data

    def __getitem__(self, index):
        lr_img = self.lr_data[index]
        hr_img = self.hr_data[index]
        lr = torch.tensor(lr_img, dtype=torch.float32).permute(2, 0, 1)  # NCHW
        hr = torch.tensor(hr_img, dtype=torch.float32).permute(2, 0, 1)

        return lr, hr

    def __len__(self):
        return len(self.hr_data)


# ============================== TRAINING ===============================

def train_loop(args, device, local_rank, checkpoint_dir, path_output, train_lr, train_hr, G, D):
    # Dataloader
    dataset = TrainData(train_lr, train_hr)
    if is_dist():
        sampler = DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank(),
                                     shuffle=True, drop_last=True)
    else:
        sampler = None

    if get_rank() == 0:
        print(f"Using {args.num_workers} data loader workers per GPU")

    num_workers = 0  # ⚠️ force single-threaded load to prevent ROCm hang
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=False,          # ROCm prefers this off unless tested
        persistent_workers=False   # must be off if num_workers=0
    )

    if get_rank() == 0:
        print(f"dataloader length: {len(loader)}")

    # Optimizers / schedulers
    g_optimizer_init = optim.Adam(G.parameters(), lr=2e-4)
    g_optimizer      = optim.Adam(G.parameters(), lr=1e-4)
    d_optimizer      = optim.Adam(D.parameters(), lr=1e-4)

    g_lr_sched_init = torch.optim.lr_scheduler.StepLR(g_optimizer_init, step_size=25, gamma=0.5)
    g_lr_sched      = torch.optim.lr_scheduler.StepLR(g_optimizer,      step_size=50, gamma=0.8)
    d_lr_sched      = torch.optim.lr_scheduler.StepLR(d_optimizer,      step_size=100, gamma=0.8)

    # Losses and wrappers
    criterion_gan      = nn.BCEWithLogitsLoss().to(device)
    criterion_content  = nn.MSELoss().to(device)
    criterion_absolute = nn.L1Loss().to(device)

    net_with_loss_init = WithLoss_init(G, criterion_content, criterion_absolute)
    # net_with_loss_D    = WithLoss_D(D, G, criterion_gan)
    # net_with_loss_G    = WithLoss_G(D, G, loss_fn1=criterion_gan, loss_fn2=criterion_content,
    #                                 loss_fn3=criterion_absolute, w1_fn1=args.w1_fn1, w2_fn2=args.w2_fn2)
    net_with_loss_D = WithLoss_D(
        D_net=D,
        loss_fn=criterion_gan
    )

    net_with_loss_G = WithLoss_G(
        D_net=D,
        loss_fn_gan=criterion_gan,
        loss_fn_content=criterion_content,
        loss_fn_abs=criterion_absolute,
        w_gan=args.w1_fn1,
        w_content=args.w2_fn2
    )

    g_init_losses, g_losses, d_losses = [], [], []
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.amp else nullcontext()

    ###################################################################
    # ---- Initial training (optional) --------------------------------------
    ###################################################################
    if args.initial_training:
        no_improve_init, min_delta_init = 5, 1e-8
        best_loss_init, wait_init = float('inf'), 0

        # Warm start if exists
        if os.path.exists(os.path.join(checkpoint_dir, 'g_init.pth')):
            target = G.module if isinstance(G, nn.parallel.DistributedDataParallel) else G
            target.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'g_init.pth'),
                                              map_location={'cuda:0': f'cuda:{local_rank}'} if is_dist() else 'cuda'))
            dist.barrier()
            if get_rank() == 0:
                print("Loaded g_init.pth")

        for epoch in range(args.n_epoch_init):
            if is_dist() and hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)

            g_sum, steps = 0.0, 0
            for lr_patch, hr_patch in loader:
                lr_patch = lr_patch.to(device, non_blocking=True)
                hr_patch = hr_patch.to(device, non_blocking=True)

                g_optimizer_init.zero_grad(set_to_none=True)
                with autocast_ctx:
                    loss = net_with_loss_init(lr_patch, hr_patch)
                
                loss.backward()
                g_optimizer_init.step()

                g_sum += float(loss.item()); steps += 1

            g_avg = g_sum / max(1, steps)
            g_init_losses.append(g_avg)
            g_lr_sched_init.step()
            if get_rank() == 0:
                print(f"[init] epoch {epoch+1}/{args.n_epoch_init} "
                        f"g_loss={g_avg:.6e}  best={best_loss_init:.6e}  wait={wait_init}",
                        flush=True)

            if g_avg < best_loss_init - min_delta_init:
                best_loss_init, wait_init = g_avg, 0
                if get_rank() == 0:
                    to_save = G.module if isinstance(G, nn.parallel.DistributedDataParallel) else G
                    torch.save(to_save.state_dict(), os.path.join(checkpoint_dir, 'g_init.pth'))
            else:
                wait_init += 1
                if get_rank() == 0:
                    print(f"[init] no improvement {wait_init}/{no_improve_init}")
                if wait_init >= no_improve_init:
                    if get_rank() == 0:
                        print("[init] early stopping")
                    break
            # Synchronize and log memory across all ranks
            if epoch == 0 and is_dist():
                dist.barrier()
                log_all_gpu_memory(device, local_rank)

    ###################################################################
    # ---- Adversarial training --------------------------------------
    ###################################################################
    no_improve_adv, min_delta_adv = 20, 1e-8
    best_g_loss, wait_adv = float('inf'), 0

    stop_training = torch.zeros(1, device=device, dtype=torch.int32)

    # Warm start for adversarial
    g_path, d_path = os.path.join(checkpoint_dir, 'g.pth'), os.path.join(checkpoint_dir, 'd.pth')
    g_init_path = os.path.join(checkpoint_dir, 'g_init.pth')

    if os.path.exists(g_path) and os.path.exists(d_path):
        if get_rank() == 0:
            print(f"[All ranks] Loading g.pth / d.pth and broadcasting...", flush=True)

        g_state = torch.load(g_path, map_location=device)
        d_state = torch.load(d_path, map_location=device)

        (G.module if is_dist() else G).load_state_dict(g_state)
        (D.module if is_dist() else D).load_state_dict(d_state)

        if get_rank() == 0:
            print("Loaded g.pth/d.pth")

    elif os.path.exists(g_init_path):
        if get_rank() == 0:
            print(f"[All ranks] Loading g_init.pth and broadcasting...", flush=True)

        g_state = torch.load(g_init_path, map_location=device)
        (G.module if is_dist() else G).load_state_dict(g_state)

        if get_rank() == 0:
            print("Loaded g_init.pth for adversarial start")

    dist.barrier()
    if get_rank() == 0:
        print(f"[All ranks synchronized after checkpoint load]", flush=True)

    
    # initialize d_loss/loss_g for first-epoch conditions
    d_loss_val, g_loss_val = 1.0, 1.0
    torch.autograd.set_detect_anomaly(False)  # Set True if you need to debug autograd issues

    def set_requires_grad(model, flag: bool):
        for p in model.parameters():
            p.requires_grad = flag
    # before training loop
    prev_g_avg = float("inf")
    prev_d_avg = float("inf")

    for epoch in range(args.n_epoch):
        if is_dist() and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        g_sum, d_sum, steps = 0.0, 0.0, 0
        g_loss_val, d_loss_val = torch.tensor(1.0), torch.tensor(1.0)  # init dummy scalars
        stop_training.zero_()

        if get_rank() == 0:
            # Example heuristic (yours, but now meaningful)
            train_G_flag = int((prev_d_avg < 0.7) or (prev_g_avg > 0.1) or (epoch == 0))
            train_D_flag = int((prev_d_avg > 0.5) or (epoch == 0))
        else:
            train_G_flag = 0
            train_D_flag = 0

        if is_dist():
            flags = torch.tensor([train_G_flag, train_D_flag], device=device, dtype=torch.int32)
            dist.broadcast(flags, src=0)
            train_G = bool(flags[0].item())
            train_D = bool(flags[1].item())
        else:
            train_G = bool(train_G_flag)
            train_D = bool(train_D_flag)

        for step, (lr_patch, hr_patch) in enumerate(loader):
            lr_patch = lr_patch.to(device, non_blocking=True)
            hr_patch = hr_patch.to(device, non_blocking=True)

            # ===========================================================
            # 0) Make fake ONCE with correct grad mode
            # ===========================================================
            if train_G:
                set_requires_grad(D, False)
                set_requires_grad(G, True)
                with autocast_ctx:
                    fake = G(lr_patch)          # grad enabled
            else:
                # IMPORTANT: no_grad so DDP doesn't expect G grads/reduction
                set_requires_grad(G, False)     # optional, but helps avoid accidental graph building
                with torch.no_grad(), autocast_ctx:
                    fake = G(lr_patch)

            # ===========================================================
            # 1️⃣  Generator training phase
            # ===========================================================

            if train_G:
                g_optimizer.zero_grad(set_to_none=True)
                with autocast_ctx:
                    g_loss_val = net_with_loss_G(hr_patch, fake)  # MUST use fake, must not call G internally
                g_loss_val.backward()
                g_optimizer.step()
            else:
                with torch.no_grad(), autocast_ctx:
                    g_loss_val = net_with_loss_G(hr_patch, fake)

            # ===========================================================
            # 2️⃣  Discriminator training phase
            # ===========================================================
            fake_d = fake.detach()
            if train_D:
                set_requires_grad(D, True)
                set_requires_grad(G, False)

                d_optimizer.zero_grad(set_to_none=True)
                with autocast_ctx:
                    d_loss_val = net_with_loss_D(hr_patch, fake_d)  # MUST not call G internally
                d_loss_val.backward()
                d_optimizer.step()
            else:
                with torch.no_grad(), autocast_ctx:
                    d_loss_val = net_with_loss_D(hr_patch, fake_d)


            # ===========================================================
            # 3️⃣  Logging
            # ===========================================================
            g_sum += float(g_loss_val.detach().item())
            d_sum += float(d_loss_val.detach().item())
            steps += 1

        # ===========================================================
        # 4️⃣  Epoch summary
        # ===========================================================
        g_avg = g_sum / max(1, steps)
        d_avg = d_sum / max(1, steps)
        g_losses.append(g_avg)
        d_losses.append(d_avg)
        prev_g_avg = g_avg
        prev_d_avg = d_avg

        if get_rank() == 0:        
            if g_avg < best_g_loss - min_delta_adv:
                best_g_loss = g_avg
                wait_adv = 0
                # Optional: save best adversarial checkpoint
                to_save_G = G.module if is_dist() else G
                to_save_D = D.module if is_dist() else D
                torch.save(to_save_G.state_dict(), g_path)
                torch.save(to_save_D.state_dict(), d_path)
            else:
                wait_adv += 1
                print(f"[adv] no improvement {wait_adv}/{no_improve_adv}", flush=True)

            print(f"[Epoch {epoch+1}/{args.n_epoch}] G_loss={g_avg:.4f}, D_loss={d_avg:.4f}, best_g_loss={best_g_loss:.4f}", flush=True)

        # save average loss history (rank 0)
        if get_rank() == 0:
            loss_data = {'g_init_losses': g_init_losses, 'g_losses': g_losses, 'd_losses': d_losses}
            with open(os.path.join(checkpoint_dir, 'avg_loss_data.json'), 'w') as f:
                json.dump(loss_data, f, indent=2)

        # Early stop decision (rank 0)
        stop_training = torch.tensor(0, device=device)
        if get_rank() == 0 and wait_adv >= no_improve_adv:
            stop_training.fill_(1)

        dist.broadcast(stop_training, src=0)
        if stop_training.item() == 1:
            break

    # Sync ranks before plots/files
    if is_dist():
        torch.distributed.barrier()

    # Save curves (rank-0)
    if get_rank() == 0:
        loss_data = {'g_init_losses': g_init_losses, 'g_losses': g_losses, 'd_losses': d_losses}
        with open(os.path.join(checkpoint_dir, 'avg_loss_data.json'), 'w') as f:
            json.dump(loss_data, f, indent=2)

        if g_init_losses:
            plt.figure(figsize=(10, 5))
            plt.plot(g_init_losses, label='G Init Losses')
            plt.title('Initial Training Loss (G)')
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True); plt.legend()
            plt.savefig(os.path.join(path_output, 'initial_training_loss.png'), dpi=300)
            plt.close()

        if g_losses and d_losses:
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax2 = ax1.twinx()
            ax1.plot(g_losses, label='G Losses')
            ax2.plot(d_losses, label='D Losses')
            fig.suptitle('Adversarial Training Losses')
            ax1.set_xlabel('Epoch'); ax1.set_ylabel('G Losses')
            ax2.set_ylabel('D Losses')
            ax1.grid(True); fig.tight_layout()
            plt.savefig(os.path.join(path_output, 'training_loss.png'), dpi=300)
            plt.close()


# ============================== EVALUATION ==============================

@torch.no_grad()
def evaluate_loop(args, device, checkpoint_dir, path_output, test_lr, loaded_scaler, G, elevation_hr=False):
    """
    Single-GPU evaluation loop.
    Loads generator weights and performs full prediction on test set.
    """
    g_adv_path = os.path.join(checkpoint_dir, 'g.pth')
    g_init_path = os.path.join(checkpoint_dir, 'g_init.pth')
    G_load = (G.module if isinstance(G, nn.parallel.DistributedDataParallel) else G)

    # # Predict with initial model if available
    # if os.path.exists(g_init_path):
    #     print("[Eval] Using g_init.pth for initial prediction...")
    #     G_load.load_state_dict(torch.load(g_init_path, map_location=device))
    #     G.eval()
    #     run_prediction(G, test_lr, loaded_scaler, path_output, 'y_pred_init.npy', device, elevation_hr)
    # else:
    #     print("[Eval] g_init.pth not found, skipping...")

    # Predict with adversarially trained model if available
    if os.path.exists(g_adv_path):
        print("[Eval] Using g.pth for adversarial prediction...")
        G_load.load_state_dict(torch.load(g_adv_path, map_location=device))
        G.eval()
        run_prediction(G, test_lr, loaded_scaler, path_output, 'y_pred.npy', device, elevation_hr)
    else:
        print("[Eval] g.pth not found, skipping...")


def run_prediction(G, test_lr, scaler, path_output, out_name, device, elevation_hr):
    """
    Run inference on the full test set (single GPU) and save results.
    """
    valid = torch.tensor(test_lr, dtype=torch.float32).permute(0, 3, 1, 2).to(device)

    bsz = 64
    outs = []
    for i in range(0, len(valid), bsz):
        out = G(valid[i:i+bsz]).cpu().numpy()
        outs.append(out)
    out = np.concatenate(outs, axis=0)

    if elevation_hr:
        out = out[:, :, :, 0]
    out = out.transpose(0, 2, 3, 1)  # NCHW → NHWC

    tt, nhr1, nhr2 = out.shape[0], out.shape[1], out.shape[2]
    yinv = scaler.inverse_transform(out.reshape(-1, out.shape[3]))
    yinv = yinv.reshape(tt, nhr1, nhr2) if yinv.ndim == 2 and yinv.shape[1] == 1 else yinv
    yinv = np.maximum(yinv, 0)

    save_path = os.path.join(path_output, out_name)
    np.save(save_path, yinv)
    print(f"[Eval] Saved prediction → {save_path} ({yinv.shape})")

# =============================== MAIN =================================

def main():
    args = build_parser().parse_args()

    num_gpus_per_node = torch.cuda.device_count()
    # print(f"num_gpus_per_node = {num_gpus_per_node}", flush=True)

    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()

    # # Each Slurm task has one GPU, and SLURM_LOCALID maps to that GPU.
    # local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    local_rank = int(rank) % int(num_gpus_per_node) # local_rank and device are 0 when using 1 GPU per task
    # local_rank = int(os.environ.get("SLURM_LOCALID", int(rank) % int(num_gpus_per_node)))
    # os.environ["HIP_VISIBLE_DEVICES"] = str(local_rank)

    # # Frontier: exactly one GPU visible per rank
    # local_rank = 0
    # torch.cuda.set_device(0)
    # device = torch.device("cuda:0")


    # Export for PyTorch DDP
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = str(args.master_addr)
    os.environ["MASTER_PORT"] = str(args.master_port)
    # os.environ["NCCL_SOCKET_IFNAME"] = "hsn0"

    # Report what this process sees
    # torch.cuda.set_device(0)   # since only 1 GPU is visible now
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.cuda.set_device(local_rank)   # ✅ correct
    device = torch.device(f"cuda:{local_rank}")

    # Enable anomaly detection (rank 0 only to avoid flood of logs)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        torch.autograd.set_detect_anomaly(True)
        print(f"[Rank {os.environ.get('RANK', 0)}] Anomaly detection enabled", flush=True)

    print(
        f"[Rank {rank}] Local rank: {local_rank}, "
        f"World size: {world_size}",
        # f"HIP_VISIBLE_DEVICES={os.getenv('HIP_VISIBLE_DEVICES')}",
        f"Selected device: {device}",
        flush=True
    )

    # Initialize the distributed process group
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{local_rank}")
    )
    dist.barrier()

    if rank == 0:
        report_topology(args, device, local_rank)


    # Paths / flags
    version = args.version
    base_dir = args.base_dir
    checkpoint_dir = os.path.join(base_dir, "models", version)
    path_output = os.path.join(base_dir, "output", version)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(path_output, exist_ok=True)

    # var = args.var.lower()

    elevation = True
    elevation_hr = False

    # Normal cached-data path
    with open(f'{checkpoint_dir}/scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)
    try:
        train_lr = read_saved_data("x_train", path_output, loaded_scaler)
        test_lr  = read_saved_data("x_test",  path_output, loaded_scaler)
        train_hr = read_saved_data("y_train", path_output, loaded_scaler)
        test_hr  = read_saved_data("y_test",  path_output, loaded_scaler)
    except FileNotFoundError as e:
        if rank == 0:
            print("Error loading cached data:", e)
        raise

    # Optional: attach elevation channels
    if elevation:
        elev_lr = np.load(f'{path_output}/elev_lr_scaled.npy')
        train_lr = np.concatenate((train_lr, elev_lr[:train_lr.shape[0]]), axis=3)
        test_lr  = np.concatenate((test_lr,  elev_lr[:test_lr.shape[0]]),  axis=3)
        if elevation_hr:
            elev_hr = np.load(f'{path_output}/elev_hr_scaled.npy')
            train_hr = np.concatenate((train_hr, elev_hr[:train_hr.shape[0]]), axis=3)
            test_hr  = np.concatenate((test_hr,  elev_hr[:test_hr.shape[0]]),  axis=3)

    #print train/test shapes
    if rank == 0:
        print(f"train_lr shape: {train_lr.shape}")
        print(f"test_lr shape:  {test_lr.shape}")
        print(f"train_hr shape: {train_hr.shape}")
        print(f"test_hr shape:  {test_hr.shape}")
    # Models
    in_channels = 2 if elevation else 1
    G = SRGAN_g_lr_26(in_channels=in_channels).to(device)
    # D = SRGAN_d_lr_odd(hr_size=train_hr[0].shape[0] * train_hr[0].shape[1]).to(device)
    hr_shape = (train_hr[0].shape[0], train_hr[0].shape[1])
    D = SRGAN_d_lr_odd(hr_size=hr_shape).to(device)


    if is_dist():
        G = nn.parallel.DistributedDataParallel(
            G, device_ids=[local_rank],
            find_unused_parameters=True
        )
        D = nn.parallel.DistributedDataParallel(
            D, device_ids=[local_rank],
            find_unused_parameters=True
        )

    # Train or Eval
    if args.mode == 'train':
        train_loop(
            args=args, device=device, local_rank=local_rank,
            checkpoint_dir=checkpoint_dir, path_output=path_output,
            train_lr=train_lr, train_hr=train_hr,
            G=G, D=D
        )
    else:
        evaluate_loop(
            args=args, device=device, local_rank=local_rank,
            checkpoint_dir=checkpoint_dir, path_output=path_output,
            test_lr=test_lr, loaded_scaler=loaded_scaler,
            G=G, elevation_hr=elevation_hr
        )
    
    # Cleanup
    if is_dist():
        torch.distributed.destroy_process_group()


# ======================================================================

if __name__ == '__main__':
    main()
