
import torch
import numpy as np


x = torch.tensor([5.5, 3])
x=x.new_ones(5,3)
print(x)

np_data=np.arange(4).reshape((2,2))
torch_data=torch.tensor(np_data,dtype=torch.float)

print(
    '\n numpy',np.matmul(np_data,np_data),
    '\n torch',torch.mm(torch_data,torch_data)
)
x= torch.rand(5, 3,requires_grad=True)
x.requires_grad_(False)


x=torch.arange(4,dtype=torch.float).reshape((2,2))
out1=x*x
out2=torch.mm(x,x)
print(
    '\n',out1,
    '\n',out2
)


x=torch.arange(6,dtype=torch.float).reshape((2,3))
y1=x.view(-1,1)
y2=x.view(3,2)
print(
    '\n',y1,
    '\n',y2
)