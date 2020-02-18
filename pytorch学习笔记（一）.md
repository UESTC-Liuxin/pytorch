[TOC]

# 前言

此为小弟pytorch的学习笔记，希望自己可以坚持下去。（2020/2/17）

[pytorch官方文档](https://pytorch.org/docs/stable/index.html)

[pytorch中文教程](https://pytorch.apachecn.org/)

# tensor

tensor是pytorch的最基本数据类型，相当于numpy中的ndarray，并且属性和numpy相似，tensor可在GPU上进行运算。

tensor常见的基本属性：

## 1.创建tensor
通常此类方法都有以下属性，特别注意，区别numpy，一般这里的入口参数size都是数字序列，而不是元组或者列表：

<img src="D:\CV\pytorch\md_img\1.1.1" alt="image-20200217184344218" style="zoom: 80%;" />


```python
#创建无初始化tensor
x1 = torch.empty(5, 3)
#创建全0值tensor
x2 = torch.zeros(5, 3, dtype=torch.long)
#创建随机值tensor
x3 = torch.rand(5, 3)
#创建全1
x4 =torch.ones(5,3)

#根据已有tensor创建新的tensor，可继承之前的所有属性，包括dtype，device等
x = x1.new_ones(5, 3, dtype=torch.double)#必须输入size
x = torch.randn_like(x1, dtype=torch.float)

#根据其余数据类型创建tensor
#列表
x=torch.tensor([1,2,3])
#ndarray
np_data=np.arange(4).reshape((2,2))
x=torch.tensor(np_data,dtype=torch.float)
```

补充：关于随机数，这里提供一个总结后的帖子：[随机数生成](https://blog.csdn.net/MarsLee_U/article/details/80549636)

## 2. 基本属性

requires_grad：是否需要梯度，如果为true则会允许自动求导

```python
#设置为有梯度属性
x= torch.rand(5, 3,requires_grad=True)
#改变梯度属性
x.requires_grad_(False)

```

## 3. 基本方法

size():获取大小

item():获取只有一个值的tensor的值

view():改变shape，此改变创建了新的副本

```python
x= torch.rand(5, 3,requires_grad=True)
#Size()为元组
x.size()
x=torch.arange(6,dtype=torch.float).reshape((2,3))

#view() 	
y1=x.view(-1,1)
y2=x.view(3,2)
print(
    '\n',y1,
    '\n',y2
)
```

## 4. 运算

tensor的所有常规运算都是基于元素的。

```python
x=torch.arange(4,dtype=torch.float).reshape((2,2))
#点乘
out1=x*x
#如果要用到叉乘
out2=torch.mm(x,x)

print(
    '\n',out1,
    '\n',out2
)
>>>>>>>
 tensor([[0., 1.],
        [4., 9.]]) 
 tensor([[ 2.,  3.],
        [ 6., 11.]])

```

## 5. GPU运算

```python
#创建tensor
x=torch.ones(3,4,dtype=torch.float)
# 当GPU可用时,我们可以运行以下代码
# 我们将使用`torch.device`来将tensor移入和移出GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # 直接在GPU上创建tensor
    # x = x.to(device)                       # 或者使用`.to("cuda")`方法
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # `.to`也能在移动时改变dtype
```

# 自动求导Autograd

对于一个tensor，若其requires_grad=True 那么它将会追踪对于该张量的所有操作。当完成计算后可以通过调用 `.backward()`，来自动计算所有的梯度。这个张量的所有梯度将会自动累加到`.grad`属性.

```python
x = torch.ones(3,3, requires_grad=True)
y=x*x+2
#
out1=y.sum()
#out1通常是一个标量，若是一个非变量，不可直接进行backward()
out1.backward()
#根据链式求导法则，x_i.grad=1*2x=2
print(
    '\n',x,
    '\n',y,
    '\n',x.grad
)
>>>>>>>>>
 tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], requires_grad=True) 
 tensor([[3., 3., 3.],
        [3., 3., 3.],
        [3., 3., 3.]], grad_fn=<AddBackward0>) 
 tensor([[2., 2., 2.],
        [2., 2., 2.],
        [2., 2., 2.]])
 
#若输出结果是一个非标量
x = torch.rand(2,2, requires_grad=True)
y = torch.rand(2,1,requires_grad=True)
out1=2*torch.mm(x,y)
#不可直接使用out1.backward()
out1.backward(torch.ones_like(y))
print(
    '\n',x,
    '\n',y,
    '\n',out1,
    '\n',y.grad
)
```

对于向量的求导，实际atuograd已经计算了雅可比矩阵，但是pytorch的内部机制，或者说本身pytorch就是为神经网络设计的，反向传播的起点为损失函数，必然为一个标量，所以，自动求导的out是需要为标量。

![image-20200218150544873](D:\CV\pytorch\md_img\2.1.1)

out1.backward(torch.ones_like(y))这里实际上输入了一个[1,1].T 的向量，代表某个标量l对out1的导数，等同于以下程序：

```python
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
```


关于更多，更深入的atuograd理解，可品读这个帖子，深入浅出：[PyTorch 自动一阶求导在标量、向量、矩阵、张量运算中的定义](https://ajz34.readthedocs.io/zh_CN/latest/ML_Notes/Autograd_Series/Autograd_TensorContract.html)

## 清空grad

当我们需要清除某个tensor的梯度时，我们需要：

```
tensor.grad = None
```

## 阻止autograd跟踪

除了修改tensor的requires_grad为False以外，我们还可以进行这样的操作，阻止autograd跟踪：

```python
with torch.no_grad():
    print((y2*y2).requires_grad):	
```

