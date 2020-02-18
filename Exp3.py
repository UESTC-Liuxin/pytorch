import torch

#情形1：

x = torch.ones(3,3, requires_grad=True)
y=x*x+2
out1=y.sum()
out1.backward()

print(
    '\n',x,
    '\n',y,
    '\n',x.grad
)

#若输出结果是一个非标量
x1 = torch.ones(2,2,requires_grad=True)
y1 = torch.ones(2,1,requires_grad=True)
out1=2*torch.mm(x1,y1)
#不可直接使用out1.backward()
out1.backward(torch.ones_like(y1))
print(
    '\n',y1.grad
)


x2 = torch.ones(2,2,requires_grad=True)
y2 = torch.ones(2,1,requires_grad=True)
z=2*torch.mm(x2,y2)
out2=z.sum()
print(out2)
#不可直接使用out1.backward()
out2.backward()
print(
    '\n',y2.grad
)
with torch.no_grad():
    print((y2*y2).requires_grad)