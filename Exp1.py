
import torch
import numpy as np

np_data=np.arange(6).reshape((2,3))
torch_data=torch.from_numpy(np_data)

print(
    '\n numpy',np_data,
    '\n torch',torch_data
)