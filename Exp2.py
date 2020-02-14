import torch
import numpy as np
from torch.autograd import Variable
#创建tensor

#
tensor0=torch.arange(4,dtype=torch.float).reshape((2,2))
#自4.0版本后variable与tensor合并
# variable= Variable(torch_data,require_grad=True)
# print(tensor0.requires_grad)
#用此函数改变tensor的requires_grad属性
tensor1=torch.arange(4,dtype=torch.float,requires_grad=True).reshape((2,2))
tensor1.requires_grad_(True)
print(tensor1.requires_grad)
#compute
out0=torch.mean(tensor0)
out1=torch.mean(tensor1)
#下面这句代码是错误的
#out0.backward()
out1.backward()
print(
    '\n',tensor0,
    '\n',tensor1,
    '\n', out0,
    '\n', out1,
    '\n', tensor0.grad,
    '\n', tensor1.grad
)
