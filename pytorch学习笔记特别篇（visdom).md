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



