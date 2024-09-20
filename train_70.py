import os
# # Set CUDA device
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import pickle
import argparse
from sklearn.preprocessing import StandardScaler
from srgan_torch import SRGAN_g, SRGAN_d, SRGAN_g_lr, SRGAN_d_lr, SRGAN_g_hr_26, SRGAN_g_hr_60_64RB, SRGAN_g_lr_smallFeature, SRGAN_d_lr_large, SRGAN_d_lr_odd, SRGAN_d_hr_odd
from dataread import daymetread
from loss_torch import WithLoss_init, WithLoss_G, WithLoss_D


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help='train, eval')
args = parser.parse_args()

# Check if CUDA is available and set the device
if torch.cuda.is_available():
    device = torch.device('cuda')
    num_gpus = torch.cuda.device_count()
    print(f"CUDA is available. Number of GPUs: {num_gpus}")
else:
    device = torch.device('cpu')
    num_gpus = 0
    print("CUDA is not available. Using CPU.")


###====================== HYPER-PARAMETERS ===========================###
batch_size = 8
n_epoch_init = 100
n_epoch = 200
# create folders to save result images and trained models
version = 'v7.0' # check the version.txt file for historical versions under output directory
save_dir = "samples"
elevation = True
elevation_hr = False
initial_training = True
readrawdata  = True
var = 'tmax'
w1_fn1=1e-4
w2_fn2=1e4

checkpoint_dir = f"models/{version}"
path_output = f'./output/{version}'
# Create directories if they don't exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(path_output, exist_ok=True)

def ReadSavedData(name, loaded_scaler):
    y_test = np.load(f'{path_output}/{name}.npy')
    scaler  = loaded_scaler
    shp = y_test.shape
    lr_test        = y_test.flatten()
    lr_test_scaled = scaler.transform(lr_test.reshape(-1,1))
    lr_test_scaled = np.reshape(lr_test_scaled,(shp[0],shp[1],shp[2],1))
    return lr_test_scaled

###====================== DATA READING ===========================###
# train_lr, test_lr, train_hr, test_hr = climateread()
if args.mode == 'train' and readrawdata:
    train_lr, test_lr, train_hr, test_hr = daymetread(path_output, checkpoint_dir, elevation, elevation_hr, Daymet_ERA5=True, high_deg=2, scaler = 'minmax', var=var)
    # Load the scaler
    with open(f'{checkpoint_dir}/scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)
else:
    # Load the scaler
    with open(f'{checkpoint_dir}/scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)
    train_lr = ReadSavedData("x_train", loaded_scaler)
    test_lr = ReadSavedData("x_test", loaded_scaler)
    train_hr = ReadSavedData("y_train", loaded_scaler)
    test_hr = ReadSavedData("y_test", loaded_scaler)
    if elevation:
        elev_scaled = np.load(f'{path_output}/elev_lr_scaled.npy')
        train_lr = np.concatenate((train_lr,elev_scaled[:np.shape(train_lr)[0]]),axis=3)
        test_lr = np.concatenate((test_lr,elev_scaled[:np.shape(test_lr)[0]]),axis=3)
        if elevation_hr:
            elev_hr_scaled = np.load(f'{path_output}/elev_hr_scaled.npy')
            train_hr = np.concatenate((train_hr,elev_hr_scaled[:np.shape(train_hr)[0]]),axis=3)
            test_hr = np.concatenate((test_hr,elev_hr_scaled[:np.shape(test_hr)[0]]),axis=3)

class TrainData(Dataset):
    def __init__(self, lr_data, hr_data):
        self.lr_data = lr_data
        self.hr_data = hr_data

    def __getitem__(self, index):
        lr_img = self.lr_data[index]
        hr_img = self.hr_data[index]
        return torch.tensor(lr_img, dtype=torch.float32).permute(2, 0, 1), torch.tensor(hr_img, dtype=torch.float32).permute(2, 0, 1)

    def __len__(self):
        return len(self.hr_data)

# Initialize models
if elevation:
    G = SRGAN_g_hr_60_64RB(in_channels=2).to(device)
else:
    G = SRGAN_g_hr_60_64RB(in_channels=1).to(device)
D = SRGAN_d_hr_odd(hr_size=train_hr[0].shape[0]*train_hr[0].shape[1]).to(device)
# input_tensor = torch.randn(1, 1, 29, 60).to(device)
# output = G(input_tensor)

# Wrap models with DataParallel if using multiple GPUs
if torch.cuda.device_count() >= 1:
    G = nn.DataParallel(G)
    D = nn.DataParallel(D)

def train():
    G.train() # set the model in training mode
    D.train()

     # Create datasets
    train_dataset = TrainData(train_lr, train_hr)
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Define optimizers
    g_optimizer_init = optim.Adam(G.parameters(), lr=2e-4)
    g_optimizer = optim.Adam(G.parameters(), lr=1e-4)
    d_optimizer = optim.Adam(D.parameters(), lr=1e-4)

    # Define learning rate schedulers
    g_lr_scheduler_init = torch.optim.lr_scheduler.StepLR(g_optimizer_init, step_size=25, gamma=0.5)
    g_lr_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=50, gamma=0.8)
    d_lr_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=100, gamma=0.8)

    # Define loss functions
    criterion_gan = nn.BCEWithLogitsLoss().to(device)  # Equivalent of sigmoid_cross_entropy
    criterion_content = nn.MSELoss().to(device)  # Mean Squared Error
    criterion_absolute = nn.L1Loss().to(device)  # Absolute Difference Error

    # Define the loss functions for initial and adversarial training
    net_with_loss_init = WithLoss_init(G, criterion_content, criterion_absolute)
    net_with_loss_D = WithLoss_D(D, G, criterion_gan)
    net_with_loss_G = WithLoss_G(D, G, loss_fn1=criterion_gan, loss_fn2=criterion_content, loss_fn3=criterion_absolute, w1_fn1=w1_fn1, w2_fn2=w2_fn2)

    g_init_losses = []
    g_losses = []
    d_losses = []

    #################################################################################
    # Early stopping settings for initial training
    #################################################################################
    if initial_training:
        no_improve_epochs_init = 5  # Number of epochs to wait before stopping if no improvement during init phase
        min_delta_init = 1e-8    # Minimum change to qualify as an improvement during init phase
        best_loss_init = float('inf')  # Track the best loss to compare against during init phase
        epochs_since_improvement_init = 0  # Track the number of epochs since last improvement during init phase

        if not readrawdata:
            if os.path.exists(os.path.join(checkpoint_dir, 'g_init.pth')):
                G.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'g_init.pth')))
                print("Pre-trained initial generator loaded!")
            else:
                print("No pre-trained generator model found.")

        # Initial training loop
        for epoch in range(n_epoch_init):
            g_loss_sum = 0
            n_steps = 0
            for lr_patch, hr_patch in train_loader:
                lr_patch = lr_patch.to(device)
                hr_patch = hr_patch.to(device)

                # Train the generator
                g_optimizer_init.zero_grad()
                loss = net_with_loss_init(lr_patch, hr_patch)
                loss.backward()
                g_optimizer_init.step()

                g_loss_sum += loss.item()
                n_steps += 1

            g_loss_avg = g_loss_sum / n_steps
            g_init_losses.append(g_loss_avg)
            print(f"Epoch [{epoch+1}/{n_epoch_init}], Avg G Loss: {g_loss_avg:.8f}")
            
            g_lr_scheduler_init.step()

            # Check for improvement
            if g_loss_avg < best_loss_init - min_delta_init:
                best_loss_init = g_loss_avg
                epochs_since_improvement_init = 0
                # Save the best model
                torch.save(G.state_dict(), os.path.join(checkpoint_dir, 'g_init.pth'))
                print(f"New best model saved with G Loss: {best_loss_init:.8f}")
            else:
                epochs_since_improvement_init += 1
                print(f"No improvement for {epochs_since_improvement_init} epochs during initialization.")
            # Early stopping
            if epochs_since_improvement_init >= no_improve_epochs_init:
                print("Early stopping triggered during initial training.")
                break

    #################################################################################
    # Adversarial learning with Early stop
    #################################################################################
    no_improve_epochs_adv = 100  # Number of epochs to wait before stopping if no improvement in adversarial phase
    min_delta_adv = 1e-8      # Minimum change to qualify as an improvement during adversarial phase
    best_g_loss_adv = float('inf')  # Track the best generator loss
    best_d_loss_adv = float('inf')  # Track the best discriminator loss
    epochs_since_improvement_adv = 0  # Track the number of epochs since last improvement in adversarial phase

    # Load pre-trained generator if not initial training
    if not readrawdata:
        if os.path.exists(os.path.join(checkpoint_dir, 'g.pth')):
            G.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'g.pth')))
            D.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'd.pth')))
            print("Pre-trained SRGAN loaded!")
        elif os.path.exists(os.path.join(checkpoint_dir, 'g_init.pth')):
            G.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'g_init.pth')))
            print("Pre-trained initial generator loaded!")
        else:
            print("No pre-trained generator model found.")

    for epoch in range(n_epoch):
        g_loss_sum = 0
        d_loss_sum = 0
        n_steps = 0
        for step, (lr_patch, hr_patch) in enumerate(train_loader):
            lr_patch = lr_patch.to(device)
            hr_patch = hr_patch.to(device)
            if epoch > 0:
                # Train Generator
                if d_loss < 0.7 or loss_g > 0.1:
                    g_optimizer.zero_grad()
                    loss_g = net_with_loss_G(lr_patch, hr_patch)
                    loss_g.backward()
                    g_optimizer.step()
                else:
                    with torch.no_grad(): # monitor g loss without train
                        loss_g = net_with_loss_G(lr_patch, hr_patch)
                # Train Discriminator
                if d_loss > 0.5:
                    d_optimizer.zero_grad()
                    loss_d = net_with_loss_D(lr_patch, hr_patch)
                    loss_d.backward()
                    d_optimizer.step()
                else:
                    with torch.no_grad(): # monitor d loss without train
                        loss_d = net_with_loss_D(lr_patch, hr_patch)
            else:
                # Train both Generator and Discriminator for the first epoch
                d_optimizer.zero_grad()
                g_optimizer.zero_grad()
                loss_g = net_with_loss_G(lr_patch, hr_patch)
                loss_g.backward()
                g_optimizer.step()
                loss_d = net_with_loss_D(lr_patch, hr_patch)
                loss_d.backward()
                d_optimizer.step()

            g_loss_sum += loss_g.item()
            d_loss_sum += loss_d.item()
            n_steps += 1

        g_loss = g_loss_sum / n_steps
        d_loss = d_loss_sum / n_steps

        if np.isnan(g_loss) or np.isnan(d_loss):
            print(f"NaN detected in losses at Epoch {epoch}, Step {step}. Stopping training.")
            break

        g_losses.append(g_loss)
        d_losses.append(d_loss)
        print(f"Epoch: [{epoch+1}/{n_epoch}], Avg G Loss: {g_loss:.8f}, Avg D Loss: {d_loss:.8f}")

        # Check for improvement in generator and discriminator
        if g_loss < best_g_loss_adv - min_delta_adv:
            best_g_loss_adv = g_loss
            torch.save(G.state_dict(), os.path.join(checkpoint_dir, 'g.pth'))
            torch.save(D.state_dict(), os.path.join(checkpoint_dir, 'd.pth'))
            print(f"New best G model saved with G Loss: {best_g_loss_adv:.8f}")
            epochs_since_improvement_adv = 0
        else:
            epochs_since_improvement_adv += 1
            print(f"No improvement for {epochs_since_improvement_adv} epochs in adversarial phase.")
        # Early stopping
        if epochs_since_improvement_adv >= no_improve_epochs_adv:
            print("Early stopping triggered during adversarial learning.")
            break

        # Update learning rates
        g_lr_scheduler.step()
        d_lr_scheduler.step()

    # save loss        
    loss_data = {'g_init_losses': g_init_losses, 'g_losses': g_losses, 'd_losses': d_losses}
    with open(f'./{checkpoint_dir}/avg_loss_data.json', 'w') as f:
        json.dump(loss_data, f)

    g_init_losses = loss_data['g_init_losses']
    g_losses = loss_data['g_losses']
    d_losses = loss_data['d_losses']
    # Plot initial training loss for the generator
    plt.figure(figsize=(10, 5))
    plt.plot(g_init_losses, label='G Init Losses', color='blue')
    plt.title('Initial Training Loss for Generator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{path_output}/initial_training_loss.png', dpi=300)
    plt.close()

    # Plot training loss for both the generator and the discriminator after initial training
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(g_losses, label='G Losses', color='blue')
    ax2.plot(d_losses, label='D Losses', color='red')
    fig.suptitle('Training Loss for Generator and Discriminator')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('G Losses', color='blue')
    ax2.set_ylabel('D Losses', color='red')
    ax1.tick_params(axis="y", labelcolor='blue')
    ax2.tick_params(axis="y", labelcolor='red')
    fig.tight_layout()
    plt.grid(True)
    plt.savefig(f'{path_output}/training_loss.png', dpi=300)
    plt.close()

def evaluate():
    
    G.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'g_init.pth')))
    G.eval()
    valid_lr_img_tensor = torch.tensor(test_lr, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # Convert to PyTorch tensor and move to device
    batch_size = 64  # Adjust batch size according to your GPU memory
    
    outputs = []
    with torch.no_grad():  # No gradient calculation for inference
        for i in range(0, len(valid_lr_img_tensor), batch_size):
            batch = valid_lr_img_tensor[i:i+batch_size]
            out = G(batch).cpu().numpy()  # Move to CPU and convert to numpy
            outputs.append(out)
    out = np.concatenate(outputs, axis=0)
    
    if elevation_hr:
        out = out[:, :, :, 0]
    # Convert from CHW to HWC
    out = out.transpose(0, 2, 3, 1)
    shp = out.shape
    tt = shp[0]
    nhr1 = shp[1]
    nhr2 = shp[2]
    y = out.flatten()
    yinv = loaded_scaler.inverse_transform(y.reshape(-1, 1))
    yinv[yinv < 0] = 0
    yinv = np.reshape(yinv, (tt, nhr1, nhr2))
    np.save(f'{path_output}/y_pred_init.npy', yinv)
    
    G.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'g.pth')))
    G.eval()
    
    valid_lr_img_tensor = torch.tensor(test_lr, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # Convert to PyTorch tensor and move to device
    batch_size = 64  # Adjust batch size according to your GPU memory
    
    outputs = []
    with torch.no_grad():  # No gradient calculation for inference
        for i in range(0, len(valid_lr_img_tensor), batch_size):
            batch = valid_lr_img_tensor[i:i+batch_size]
            out = G(batch).cpu().numpy()  # Move to CPU and convert to numpy
            outputs.append(out)
    
    out = np.concatenate(outputs, axis=0)
    
    if elevation_hr:
        out = out[:, :, :, 0]
    
    # Convert from CHW to HWC
    out = out.transpose(0, 2, 3, 1)
    shp = out.shape
    tt = shp[0]
    nhr1 = shp[1]
    nhr2 = shp[2]
    y = out.flatten()
    yinv = loaded_scaler.inverse_transform(y.reshape(-1, 1))
    yinv[yinv < 0] = 0
    yinv = np.reshape(yinv, (tt, nhr1, nhr2))
    np.save(f'{path_output}/y_pred.npy', yinv)


if __name__ == '__main__':

    if args.mode == 'train':
        train()
    elif args.mode == 'eval':
        evaluate()
    else:
        raise Exception("Unknow --mode")