# coding=utf-8
import time
from visdom import Visdom
import requests
import os
import numpy as np

viz = Visdom(server='http://[::1]', port=8097,env='test')
assert viz.check_connection()

# 视频下载可能比较慢，耐心等几分中
video_file = "./DATA//trailer.ogv"
# if not os.path.exists(video_file):
#     video_url = 'http://media.w3.org/2010/05/sintel/trailer.ogv'
#     res = requests.get(video_url)
#     with open(video_file, "wb") as f:
#         f.write(res.content)
# with open(video_file, "wb") as f:
#     f.write(res.content)
viz.video(
    videofile=video_file,
    win='win_video'
)

# 图片
# 单张图片
viz.image(
    np.random.rand(3, 256, 256), #格式 C*H*W
    win='win_image',
    opts={
        'title': 'single_image',
        'showlegend': True
    }
)
# 多张图片
viz.images(
    np.random.rand(20, 3, 64, 64),
    win='win_images',
    opts={
        'title': 'multi-images',
    }
)

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

# time.sleep(5)
# 更新样式
viz.update_window_opts(
    win=old_scatter,
    opts={
        'title': 'New Scatter',
        'legend': ['Apple', 'Banana'],
        'markersymbol': 'dot'
    }
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
