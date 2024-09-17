import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

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
    # Convert data to tensor if it's a numpy array
    if dist.get_rank() == src:
        if isinstance(data, np.ndarray):
            data_tensor = torch.tensor(data, dtype=torch.float32)
        else:
            data_tensor = data
        shape = torch.tensor(data_tensor.shape, dtype=torch.long)
    else:
        data_tensor = None
        shape = torch.zeros(2, dtype=torch.long)  # Assuming 2D data, adjust if needed

    # Broadcast the shape
    dist.broadcast(shape, src=src)

    # For non-source ranks, initialize an empty tensor to hold the data
    if data_tensor is None:
        data_tensor = torch.empty(tuple(shape.tolist()), dtype=torch.float32)

    # Broadcast the actual data tensor
    dist.broadcast(data_tensor, src=src)

    return data_tensor

# def test_broadcast():
    setup_distributed()

    # Only rank 0 has the data initially
    if dist.get_rank() == 0:
        print(f"Rank 0: Broadcasting data...")
        data = np.random.rand(5, 5)  # Create random data (numpy array)
        print(f"Original Data on Rank 0:\n{data}")
    else:
        data = None  # Other ranks don't have data initially

    # Broadcast the data from rank 0 to all ranks
    broadcasted_data = broadcast_data(data, src=0)

    # Print the received data on each rank
    print(f"Rank {dist.get_rank()} received data:\n{broadcasted_data.numpy()}")

    dist.barrier()
    dist.destroy_process_group()

def train():
    # Initialize distributed setup
    setup_distributed()

    # Only rank 0 reads the data initially
    if dist.get_rank() == 0:
        # Create some dummy data (100 samples, 10 features each)
        train_data = np.random.rand(100, 10)
        train_labels = np.random.rand(100, 1)
    else:
        train_data, train_labels = None, None

    # Broadcast data to all ranks
    train_data = broadcast_data(train_data)
    train_labels = broadcast_data(train_labels)
    # Print the received data on each rank
    print(f"Rank {dist.get_rank()} received data")

    # Create a dataset and dataloader from the broadcasted data
    dataset = TensorDataset(train_data, train_labels)
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