import torch
import torch.distributed as dist
import numpy as np
import os

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

def test_broadcast():
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

if __name__ == "__main__":
    test_broadcast()
