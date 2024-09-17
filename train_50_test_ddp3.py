import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import pickle

# Replace this with the actual function to read saved data
def ReadSavedData(name, scaler, path_output):
    y_test = np.load(f'{path_output}/{name}.npy')
    shp = y_test.shape
    lr_test = y_test.flatten()
    lr_test_scaled = scaler.transform(lr_test.reshape(-1, 1))
    lr_test_scaled = np.reshape(lr_test_scaled, (shp[0], shp[1], shp[2], 1))
    return lr_test_scaled

# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

def setup_distributed():
    """
    Initialize the process group and set up distributed environment variables.
    """
    if 'RANK' not in os.environ:
        os.environ['RANK'] = os.environ.get('SLURM_PROCID', '0')
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = os.environ.get('SLURM_NTASKS', '1')
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = os.environ.get('SLURM_LOCALID', '0')

    dist.init_process_group(backend='gloo', init_method='env://')

def broadcast_data(data, src=0):
    """
    Broadcast data from the source rank (src) to all other ranks. 
    """
    if dist.get_rank() == src:
        if isinstance(data, np.ndarray):
            data_tensor = torch.tensor(data, dtype=torch.float32)
        else:
            data_tensor = data
        shape = torch.tensor(data_tensor.shape, dtype=torch.long)
    else:
        data_tensor = None
        shape = torch.zeros(4, dtype=torch.long)  # Adjust shape depending on the dimensions of your data

    # Broadcast the shape
    dist.broadcast(shape, src=src)

    # For non-source ranks, initialize an empty tensor to hold the data
    if data_tensor is None:
        data_tensor = torch.empty(tuple(shape.tolist()), dtype=torch.float32)

    # Broadcast the actual data tensor
    dist.broadcast(data_tensor, src=src)

    return data_tensor

def train():
    setup_distributed()

    # Data paths
    checkpoint_dir = 'models/v5.0'
    path_output = 'output/v5.0'
    elevation = True
    elevation_hr = False


    # Load the scaler
    with open(f'{checkpoint_dir}/scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)

    # Read saved data (replace with actual file names and scaler)
    train_lr = ReadSavedData("x_train", loaded_scaler, path_output)
    test_lr = ReadSavedData("x_test", loaded_scaler, path_output)
    train_hr = ReadSavedData("y_train", loaded_scaler, path_output)
    test_hr = ReadSavedData("y_test", loaded_scaler, path_output)

    # Add elevation if required
    if elevation:
        elev_scaled = np.load(f'{path_output}/elev_lr_scaled.npy')
        train_lr = np.concatenate((train_lr, elev_scaled[:np.shape(train_lr)[0]]), axis=3)
        test_lr = np.concatenate((test_lr, elev_scaled[:np.shape(test_lr)[0]]), axis=3)
        if elevation_hr:
            elev_hr_scaled = np.load(f'{path_output}/elev_hr_scaled.npy')
            train_hr = np.concatenate((train_hr, elev_hr_scaled[:np.shape(train_hr)[0]]), axis=3)
            test_hr = np.concatenate((test_hr, elev_hr_scaled[:np.shape(test_hr)[0]]), axis=3)

    # Convert broadcasted data to PyTorch tensors
    train_lr = torch.tensor(train_lr, dtype=torch.float32)
    train_hr = torch.tensor(train_hr, dtype=torch.float32)
    test_lr = torch.tensor(test_lr, dtype=torch.float32)
    test_hr = torch.tensor(test_hr, dtype=torch.float32)

    print(f"Rank {dist.get_rank()} has the data")

    # Create a dataset and dataloader from the broadcasted data
    dataset = TensorDataset(train_lr, train_hr)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Set up model and optimizer
    local_rank = int(os.environ['LOCAL_RANK'])
    model = SimpleModel().to(local_rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Training loop (only a few epochs for testing)
    model.train()
    for epoch in range(2):
        epoch_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(local_rank), target.to(local_rank)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Rank {dist.get_rank()}, Epoch [{epoch+1}], Loss: {epoch_loss/len(train_loader)}")

    # Clean up distributed processes
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    train()
