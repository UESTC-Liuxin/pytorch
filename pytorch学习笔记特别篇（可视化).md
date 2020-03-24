[TOC]



# visdom学习笔记

## visdom介绍

Visdom是Facebook专门为PyTorch开发的一款可视化工具，其开源于2017年3月。Visdom十分轻量级，但却支持非常丰富的功能，能胜任大多数的科学运算可视化任务，支持远程，支持tensor和numpy数据格式。[中文教程](https://ptorch.com/news/77.html)

![界面效果图：](https://camo.githubusercontent.com/2b1b3f8ceb9b9379d59183352f1ca3f2e6bbf064/68747470733a2f2f6c68332e676f6f676c6575736572636f6e74656e742e636f6d2f2d6833487576625532563053666771675847694b334c50676845357671765330707a704f6253305967475f4c41424d466b36324a4361334b56755f324e565f344c4a4b614161352d74673d7330)

## visdom入口参数

```python
viz = Visdom(server，...)
```

- `server`:visdom服务器的主机名(默认值:`'http://localhost'`)
- `port`:visdom服务器的端口(默认值:`8097`)
- `base_url`:基本visdom服务器url(默认:`/`)
- `env`:没有提供`env`时要绘制的默认环境(默认是`main`)
- `raise_exceptions`:失败时引发异常，而不是打印它们(默认值:”True”(性能好))
- `log_to_filename`:如果不是全部，则将所有绘图和更新事件记录到给定文件中(附加模式)，以便以后可以使用`replay_log`来重放它们(默认值:”None”)
- `use_incoming_socket`:启用套接字以从Web客户端接收事件，允许用户注册回调(默认值:”True”)
- `http_proxy_host`:已弃用。使用Proxies参数可获得完整的代理支持。
- `http_proxy_port`:已弃用。使用Proxies参数可获得完整的代理支持。
- `username`:用于验证的用户名，如果服务器以`-enable_login`开头(默认值:`None`)
- `password`:用于验证的密码，如果服务器以`-enable_login`开头(默认:`None`)
- `proxies`:字典映射协议，用于在每个请求上使用的代理URL(例如{`http`:`foo.bar:3128`})。(默认值:”无”)
- `offline`:标记为在脱机模式下运行visdom，在该模式下，所有请求都记录到文件而不是服务器上。需要设置”log_to_filename”。在离线模式下，所有不创建或更新绘图的visdom命令都将简单地返回True。(默认值:”False”

## envs

环境（envs）是visdom当中常用的最大的一个作用域，在一个envs中可以装载多个Panes，每个pane都可以绘制图表，图片，以及流媒体视频。

```python
viz = Visdom(server='http://127.0.0.1', port=8097,env='test')
assert viz.check_connection()
```

server默认为本机的环回地址，IPV4:127.0.0.1 ;IPV6:[::1],对应的就是localhost的本地主机名，同时也是缺省值。通常会在后面添加连接失败报错。

## pane

pane是绘图实体，由属性win指定。

### 基本

`Visdom`提供以下基本的可视化功能：

- [vis.image](https://ptorch.com/news/77.html#visimage):图片
- [vis.images](https://ptorch.com/news/77.html#visimages):图像列表
- [vis.text](https://ptorch.com/news/77.html#vistext):任意的HTML
- [vis.video](https://ptorch.com/news/77.html#visvideo):视频
- [vis.svg](https://ptorch.com/news/77.html#vissvg):SVG对象
- [vis.save](https://ptorch.com/news/77.html#vissave):序列化状态服务器端

### 绘制

我们已经封装了几种常见的绘图类型，以便轻松创建基本可视化。这些可视化由[Plotly](https://plot.ly/)提供[支持](https://plot.ly/)。

目前支持以下`API`：

- [vis.scatter](https://ptorch.com/news/77.html#visscatter):2D或3D散点图
- [vis.line](https://ptorch.com/news/77.html#visline):线图
- [vis.updateTrace](https://ptorch.com/news/77.html#visupdatetrace):更新现有的线/散点图
- [vis.stem](https://ptorch.com/news/77.html#visstem):干地块
- [vis.heatmap](https://ptorch.com/news/77.html#visheatmap):热图地块
- [vis.bar](https://ptorch.com/news/77.html#visbar):条形图
- [vis.histogram](https://ptorch.com/news/77.html#vishistogram):直方图
- [vis.boxplot](https://ptorch.com/news/77.html#visboxplot):盒子
- [vis.surf](https://ptorch.com/news/77.html#vissurf):表面
- [vis.contour](https://ptorch.com/news/77.html#viscontour):等高线图
- [vis.quiver](https://ptorch.com/news/77.html#visquiver):这个应该同matplotlib相似，是用于画场图，矢量方向图的
- [vis.mesh](https://ptorch.com/news/77.html#vismesh):网格图

绘图功能采用可选opts表格作为输入，可用于更改绘图属性（通用或特定于绘图）的属性。所有输入参数都在一个表中指定; 输入参数是基于它们在输入表中具有的键的匹配。

以下opts是通用的，它们对于所有可视化（除了plot.image和plot.text）是相同的：

- opts.title ：图标题
- opts.width ：图宽度
- opts.height ：身高
- opts.showlegend ：显示图例（true或false）
- opts.xtype ：x轴的类型（'linear'或'log'）
- opts.xlabel ：x轴的标签
- opts.xtick ：在x轴上显示刻度（boolean）
- opts.xtickmin ：先在x轴上打勾（number）
- opts.xtickmax ：在x轴上的最后一个勾号（number）
- opts.xtickvals ：在x轴（蜱位置table的number多个）
- opts.xticklabels：蜱上x轴（标签table的string多个）
- opts.xtickstep ：x轴上的滴答声之间的距离（number）
- opts.ytype ：y轴的类型（'linear'或'log'）
- opts.ylabel ：y轴的标签
- opts.ytick ：在y轴上显示刻度（boolean）
- opts.ytickmin ：首先在y轴上打勾（number）
- opts.ytickmax ：最后在y轴上打勾（number）
- opts.ytickvals ：在y轴的刻度位置（table的number多个）
- opts.yticklabels：蜱上y轴标签（table的string多个）
- opts.ytickstep ：Y轴上的刻度之间的距离（number）
- opts.marginleft ：左边距（以像素为单位）
- opts.marginright ：右边距（以像素为单位）
- opts.margintop ：顶部边距（以像素为单位）
- opts.marginbottom：底部边距（以像素为单位）

其他选项是可视化特定的，并在功能的文档中进行了描述。

## 具体示例

### 图片

```python
# 图片
# 单张图片
viz.image(
    np.random.rand(3, 512, 256),
    opts={
        'title': 'Random',
        'showlegend': True
    }
)
# 多张图片
viz.images(
    np.random.rand(20, 3, 64, 64),
    opts={
        'title': 'multi-images',
    }
)

```

### 散点图

此功能绘制2D或3D散点图。它需要输入一个==Nx2或 一个Nx3张量X==来指定N散点图中点的位置。一个可选的N张量，Y其中包含离散的标签，范围介于1和K可以指定 - 标签将反映在标记的颜色。以下opts是支持的：

- opts.colormap ：colormap（string; default = 'Viridis'）
- ops.textlabels : 单个数据点文本备注
- opts.markersymbol：标志符号（string;默认= 'dot'）
- opts.markersize ：标记大小（number;默认= '10'）
- opts.markercolor：每个标记的颜色。（torch.*Tensor; default = nil）
- opts.legend ：table包含图例名称
- opts.markercolor是一个整数值的张量。张量可以是大小N或N x 3或K或K x 3。

	- 尺寸张量N：每个数据点的单一强度值。0 =黑色，255 =红色
	- 尺寸张量N x 3：每个数据点的红色，绿色和蓝色强度。0,0,0 =黑色，255,255,255 =白色
	- 尺寸K和张量K x 3：与每个数据点具有唯一的颜色不同，对于特定标签的所有点共享相同的颜色。

```PYTHON
# 散点图
X=np.random.rand(10, 2) * 100
Y = np.random.rand(10)
Y = (Y + 1.5).astype(int),  # 100个标签1和2
old_scatter = viz.scatter(
    X=X,
    Y=Y,
    win='win_scatter2D',
    opts={
        'title': '2D_Scatter',
        'legend': ['A', 'B'],
        'xlabel':'x1',
        'textlabels':['%d' % i for i,value in enumerate(X)],
        # 'xtickmin': 0,
        # 'xtickmax': 100,
        # 'ytickmin': 0,
        # 'ytickmax': 100,
        'markersymbol': 'cross-thin-open',
        'width': 400,
        'height': 300
    },
)

X=np.random.rand(10, 3)
# 3D散点图
viz.scatter(
    X=X,
    Y=Y,
    win='win_scatter3D',
    opts={
        'title': '3D Scatter',
        'legend': ['Men', 'Women'],
        'markersize': 5,
        'textlabels':['%d' % i for i,value in enumerate(X)],
        'xlabel':'x',
        'ylabel':'y',
        'zlabel':'z'
    }
)
```

### 直方图

该功能绘制指定数据的直方图。它需要输入一个N张量X来指定构建直方图的数据。

opts目前支持以下特定于图表的功能：

- opts.numbins：垃圾箱数量（number;默认= 30）

```python
# 直方图
viz.histogram(
    X=np.random.randint(255,size=(100)),
    win='win_histogram',
    opts=dict(
        xlabel='gray_rank',
        ylabel='count',
        title='histogram',
        numbins=10   #利用numbins对数据单位进行划分，默认是30组
    )
)
```



### 连线图

用来画线图。它需要一个形状为`N`或者`N×M`的tensor `Y`，用来指定 `M`条线的值(每条线上有`N`个点)。和一个可供选择的tensor `X` 用来指定对应的 x轴的值; `X`可以是一个长度为`N`的tensor（这种情况下，M条线共享同一个x轴），也可以是形状和`Y`一样的tensor。==遗憾的是，visdom不支持3d线图==

支持以下特定选项：

- `options.fillarea` : 填充线下面的区域 (`boolean`)

- `options.colormap` : 色图 (`string`; default = `'Viridis'`)

- `options.markers` : 显示点标记 (`boolean`; default = `false`)

- `options.markersymbol`: 标记的形状 (`string`; default = `'dot'`)

- `options.markersize` : 标记的大小 (`number`; default = `'10'`)

- `options.legend` : 保存图例名字的 `table`

```python
  #连线图
  viz.line(
      Y=X,
      X=np.arange(10),
      win='line_3D',
      opts=dict(
          legend=['a','b','c'],
          fillarea=True,
          markers=True,
          markersymbol='dot'
  
      )
)
```

  

### 柱状图

此功能可以画正常的，堆起来的，或分组的的条形图。 输入参数：

- X(tensor):形状 N 或 N×M，指定每个条的高度。如果X有M列，那么每行的值可以看作一组或者把他们值堆起来（取决与options.stacked是否为True）。
- Y(tensor, optional):形状 N，指定对应的x轴的值。

支持以下特定选项：

- options.columnnames: table containing x-axis labels
- options.stacked : stack multiple columns in X
- options.legend : table containing legend labels

```python
# 柱状图
viz.bar(
    X=np.random.rand(20),
    win='bar0',
)
viz.bar(
    X=np.abs(np.random.rand(5, 3)),  # 5个列，每列有3部分组成
    win='bar1',
    opts={
        'stacked': True,
        'legend': ['A', 'B', 'C'],
        'rownames': ['2012', '2013', '2014', '2015', '2016']
    }
)

viz.bar(
    X=np.random.rand(20, 3),
    win='bar2',
    opts={
        'stacked': False,
        'legend': ['America', 'Britsh', 'China']
    }
)

```

### 热力图，地理图，表面图

```python
# 热力图，地理图，表面图
x = np.tile(np.arange(1, 101), (100, 1))
y = x.transpose()
X = np.exp((((x - 50) ** 2) + ((y - 50) ** 2)) / -(20.0 ** 2))
viz.heatmap(
    X=X,
    win='heatmap',
    opts=dict(
        colormap= 'Hot'
    )
)

# 地表图
viz.contour(
    X=X,
    win='contour',
    opts=dict(
        colormap='Viridis'
    )
)

# 表面图
viz.surf(
    X=X,
    win='surf',
    opts=dict(
        colormap='Hot'
    )
)

```

[完整代码](https://github.com/UESTC-Liuxin/pytorch/blob/master/visdom_test.py)

# tensorboardX学习笔记

ensorFlow 附加的工具Tensorboard 是一个很好用的视觉化工具。他可以记录数字，影像或者是声音资讯，对于观察类神经网路训练的过程非常有帮助。tensorboardX 的目的就是让其他tensorboard 的功能都可以轻易的被非TensorFlow 的框架使用。官方教程：https://tensorboardx.readthedocs.io/en/latest/tensorboard.html


## 安装

```bash
pip install tensorboardX
pip install tensorflow
```

## 初始化

tensorboard所有操作都是基于SummaryWriter类，以及类中的方法

```python
from tensorboardX import SummaryWriter
writer = SummaryWriter()
```
**SummaryWriter主要方法**

**标量**

- add_scalar(self, tag, scalar_value, global_step=None, walltime=None)
- add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None)
- export_scalars_to_json(self, path)
- add_custom_scalars_multilinechart(self, tags, category='default', title='untitled')
- add_custom_scalars_marginchart(self, tags, category='default', title='untitled')
- add_custom_scalars(self, layout)

**param**

`add_hparams`**(***hparam_dict=None***,** *metric_dict=None***,** *name=None***,** *global_step=None***)**

```python
from tensorboardX import SummaryWriter
with SummaryWriter() as w:
    for i in range(5):
        w.add_hparams({'lr': 0.1*i, 'bsize': i},
                      {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})
```

**图像图表**

- add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None)

  ==记录直方图很耗CPU 资源，不要常用==

- add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
  
-  add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW')
   
- add_image_with_boxes(self, tag, img_tensor, box_tensor, global_step=None,walltime=None, dataformats='CHW', **kwargs)**
  
- 将matplotlib图形渲染成图像并将其添加到摘要中add_figure(self, tag, figure, global_step=None, close=True, walltime=None)
  
-  add_graph(self, model, input_to_model=None, verbose=False, **kwargs)
   
- add_pr_curve(self, tag, labels, predictions, global_step=None,num_thresholds=127, weights=None, walltime=None)pr：precision-recall
  
- add_pr_curve_raw(self, tag, true_positive_counts  , false_positive_counts, true_negative_counts,  false_negative_counts,precision,  recall,global_step=None,num_thresholds=127,weights=None, walltime=None)
  
- `add_image_with_boxes`**(***tag***,** *img_tensor***,** *box_tensor***,** *global_step=None***,** *walltime=None***,** *dataformats='CHW'***,** *labels=None***,** ***kwargs***)**==牛逼==

**多媒体**

- add_video(self, tag, vid_tensor, global_step=None, fps=4, walltime=None)
  
- add_audio(self, tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None)
  
- add_text(self, tag, text_string, global_step=None, walltime=None)
  
- add_embedding(self, mat, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None)
  
    
  
## 主要参数

- tag： 标签，数据标识符号，名称(string):

- main_tag：标签组名称(string)

- tag_scalar_dict：标签键值对,（dict）；

- scalar_value : 要保存的值(float or string/blobname)

- global_step : 要记录的全部步长值(int)

- walltime (float): Optional override default walltime (time.time()) of event可选覆盖默认的

- walltime（time.time（）），以秒为单位事件的时期

- dataformats：CHW（默认），NCHW（默认），HW，NHWC

- img_tensor (torch.Tensor, numpy.array, or string/blobname): 默认shape为（3，H，W）。可以使用

  ==torchvision.utils.make_grid（）==将一批张量转换为3xHxW格式或调用add_images完成。也可以设置为

  （1，H，W）、（H，W）。为避免出错，建议使用默认值CHW（channel在前，更符合一般习惯），

- img_tensor需要和dataformats匹配。

- model (torch.nn.Module): ==添加模型图graph==

- input_to_model (torch.Tensor or list of torch.Tensor): 喂入模型的数据.

## Example

```python
# demo.py
import os
import time
import torch
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
from torchvision import datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

TIMESTAMP = time.strftime("%Y-%m-%d-%H-%M-%S")
log_dir = 'logs/test/' + TIMESTAMP
resnet18 = models.resnet18(False)
writer = SummaryWriter(log_dir=log_dir)
writer.flush()
sample_rate = 44100
freqs = [262, 294, 330, 349, 392, 440, 440, 440, 440, 440, 440]
# with SummaryWriter() as w:
#     for i in range(5):
#         w.add_hparams({'lr': 0.1*i, 'bsize': i},
#                       {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})
for n_iter in range(100):

    dummy_s1 = torch.rand(1)
    dummy_s2 = torch.rand(1)
    # data grouping by `slash`
    writer.add_scalar('data/scalar1', dummy_s1[0], n_iter)
    writer.add_scalar('data/scalar2', dummy_s2[0], n_iter)

    writer.add_scalars('data/scalar_group', {'xsinx': n_iter * np.sin(n_iter),
                                             'xcosx': n_iter * np.cos(n_iter),
                                             'arctanx': np.arctan(n_iter)}, n_iter)

    dummy_img = torch.rand(32, 3, 64, 64)  # output from network
    if n_iter % 10 == 0:
        x = vutils.make_grid(dummy_img, normalize=True, scale_each=True)
        writer.add_image('Image', x, n_iter)

        dummy_audio = torch.zeros(sample_rate * 2)
        for i in range(x.size(0)):
            # amplitude of sound should in [-1, 1]
            dummy_audio[i] = np.cos(freqs[n_iter // 10] * np.pi * float(i) / float(sample_rate))
        writer.add_audio('myAudio', dummy_audio, n_iter, sample_rate=sample_rate)

        writer.add_text('Text', 'text logged at step:' + str(n_iter), n_iter)

        # for name, param in resnet18.named_parameters():
        #     writer.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)

        # needs tensorboard 0.4RC or later
        writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(100), n_iter)


dataset = datasets.MNIST('mnist', train=False, download=True)
images = dataset.test_data[:100].float()
label = dataset.test_labels[:100]
print(images.size())
print(label.size())
features = images.view(100, 784)
writer.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))
#
writer.add_image_with_boxes('CoCo',torch.rand(3,128,128),torch.tensor([[10,10,50,50]]))
# export scalar data to JSON for external processing
writer.export_scalars_to_json("./all_scalars.json")
writer.close()

```

