
import torch
import numpy as np

np_data=np.arange(4).reshape((2,2))
torch_data=torch.FloatTensor(np_data)

print(
    '\n numpy',np.matmul(np_data,np_data),
    '\n torch',torch.mm(torch_data,torch_data)
)