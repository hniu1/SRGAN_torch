import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def setup_distributed():
    # Manually set RANK, WORLD_SIZE, and LOCAL_RANK if not provided by SLURM
    if 'RANK' not in os.environ:
        os.environ['RANK'] = os.environ.get('SLURM_PROCID', '0')
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = os.environ.get('SLURM_NTASKS', '1')
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = os.environ.get('SLURM_LOCALID', '0')

    print(f"RANK: {os.environ['RANK']}")
    print(f"WORLD_SIZE: {os.environ['WORLD_SIZE']}")
    print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
    print(f"MASTER_PORT: {os.environ['MASTER_PORT']}")
    print(f"LOCAL_RANK: {os.environ['LOCAL_RANK']}")

    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://')

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

def main():
    # Setup distributed environment
    local_rank = setup_distributed()
    global_rank = dist.get_rank()

    # Create model and wrap it with DistributedDataParallel
    model = SimpleModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Create a simple dataset and DataLoader
    dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 10))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Define optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(2):  # Run 2 epochs
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(local_rank), target.to(local_rank)

            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0 and global_rank == 0:  # Only print from rank 0
                print(f"Epoch [{epoch + 1}/2], Batch [{batch_idx}], Loss: {loss.item()}")

    # Cleanup the distributed environment
    cleanup()

if __name__ == '__main__':
    main()
