import matplotlib.pyplot as plt
import os
import json

version = 'dy_v0.1'  # Replace with actual version
base_dir = os.environ.get("SRGAN_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
checkpoint_dir = f'{base_dir}/models/{version}'  # Replace with actual path
path_output = f'{base_dir}/output/{version}'  # Replace with actual path

# load loss data from json
with open(os.path.join(checkpoint_dir, 'avg_loss_data.json'), 'r') as f:
    loss_data = json.load(f)

g_init_losses = loss_data.get('g_init_losses', [])
g_losses = loss_data.get('g_losses', [])
d_losses = loss_data.get('d_losses', [])

if g_init_losses:
    plt.figure(figsize=(10, 5))
    plt.plot(g_init_losses, label='G Init Losses')
    plt.title('Initial Training Loss (G)')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True); plt.legend()
    plt.savefig(os.path.join(path_output, 'initial_training_loss.png'), dpi=300)
    plt.close()

if g_losses and d_losses:
    # fig, ax1 = plt.subplots(figsize=(10, 5))
    # ax2 = ax1.twinx()
    # ax1.plot(g_losses, label='G Losses', color='blue')
    # ax2.plot(d_losses, label='D Losses', color='red')
    # fig.suptitle('Adversarial Training Losses')
    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('G Losses', color='blue')
    # ax2.set_ylabel('D Losses', color='red')
    # ax1.tick_params(axis="y", labelcolor='blue')
    # ax2.tick_params(axis="y", labelcolor='red')
    # ax1.grid(True)
    fig = plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='G Losses', color='blue')
    plt.plot(d_losses, label='D Losses', color='red')
    plt.title('Adversarial Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(path_output, 'training_loss.png'), dpi=300)
    plt.close()