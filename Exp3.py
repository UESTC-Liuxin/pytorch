import torch
# #创建tensor
# x=torch.ones(2,2,requires_grad=True)
# print(x)
# y = x + 2
# print(y)
# z = y * y * 3  #torch中的乘法是对应元素相乘
# out = z.mean()
# print(z, out)
# out.backward()
# print(x.grad)


x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(x,y)


