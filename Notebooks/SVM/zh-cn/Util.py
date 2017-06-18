import numpy as np
import matplotlib.pyplot as plt

from math import pi

np.random.seed(142857)

# 生成简单的测试数据集
def gen_two_clusters(size=100, center=0, scale=1, dis=2):
    center1 = (np.random.random(2) + center - 0.5) * scale + dis
    center2 = (np.random.random(2) + center - 0.5) * scale - dis
    cluster1 = (np.random.randn(size, 2) + center1) * scale
    cluster2 = (np.random.randn(size, 2) + center2) * scale
    data = np.vstack((cluster1, cluster2)).astype(np.float32)
    labels = np.array([1] * size + [-1] * size)
    indices = np.random.permutation(size * 2)
    data, labels = data[indices], labels[indices]
    return data, labels

# 生成螺旋线数据集
def gen_spiral(size=50, n=4, scale=2):
    xs = np.zeros((size * n, 2), dtype=np.float32)
    ys = np.zeros(size * n, dtype=np.int8)
    for i in range(n):
        ix = range(size * i, size * (i + 1))
        r = np.linspace(0.0, 1, size+1)[1:]
        t = np.linspace(2 * i * pi / n, 2 * (i + scale) * pi / n, size) + np.random.random(size=size) * 0.1
        xs[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        ys[ix] = 2 * (i % 2) - 1
    return xs, ys

# 画出决策边界；如果只关心算法本身，可以略去这一段代码不看
def visualize2d(clf, x, y, draw_background=False):
    axis, labels = np.array(x).T, np.array(y)
    decision_function = lambda xx: clf.predict(xx)

    nx, ny, padding = 400, 400, 0.2
    x_min, x_max = np.min(axis[0]), np.max(axis[0])
    y_min, y_max = np.min(axis[1]), np.max(axis[1])
    x_padding = max(abs(x_min), abs(x_max)) * padding
    y_padding = max(abs(y_min), abs(y_max)) * padding
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding

    def get_base(nx, ny):
        xf = np.linspace(x_min, x_max, nx)
        yf = np.linspace(y_min, y_max, ny)
        n_xf, n_yf = np.meshgrid(xf, yf)
        return xf, yf, np.c_[n_xf.ravel(), n_yf.ravel()]

    xf, yf, base_matrix = get_base(nx, ny)
    z = decision_function(base_matrix).reshape((nx, ny))
    
    labels[labels == -1] = 0
    n_label = len(np.unique(labels))
    xy_xf, xy_yf = np.meshgrid(xf, yf, sparse=True)
    colors = plt.cm.rainbow([i / n_label for i in range(n_label)])[labels]

    plt.figure()
    if draw_background:
        plt.pcolormesh(xy_xf, xy_yf, z, cmap=plt.cm.Paired)
    else:
        plt.contour(xf, yf, z, c='k-', levels=[0])
    plt.scatter(axis[0], axis[1], c=colors)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

