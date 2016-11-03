import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sigma = 1
lb = 0.1

mu = np.array([[(1, 1), (-1, -1), (2, 2)]])

alpha = np.array([6, 6, 1])
b = -8


def gaussian_kernel(x):
    return np.exp(-np.sum(np.array(x - mu) ** 2, axis=2) * 0.5 / sigma ** 2)


def main():

    x = np.array([[(-1, -1), (1, 1), (2, 2), (1, -1), (-1, 1), (-2, 2)]])
    y = np.array([-1, -1, -1, 1, 1, 1])
    # x = np.array([[(0, 1), (1, 1), (0, 0), (1, 0)]])
    # y = np.array([-1, 1, 1, -1])

    def f(_x):
        return np.sum(alpha * gaussian_kernel(_x), axis=1) + b

    plot_scale = 2
    plot_num = 100
    x_min, x_max = min(-1, np.min(x)), max(1, np.max(x))

    xf = np.linspace(x_min * plot_scale, x_max * plot_scale, plot_num)
    yf = np.linspace(x_min * plot_scale, x_max * plot_scale, plot_num)
    x_base, y_base = np.meshgrid(xf, yf)
    base_matrix = np.dstack((x_base, y_base)).reshape((plot_num * plot_num, 1, 2))
    ans = f(base_matrix).reshape((plot_num, plot_num))

    plt.contourf(x_base, y_base, ans, cmap=cm.Spectral)
    plt.scatter(x[0][:, 0], x[0][:, 1], c=y, s=40, cmap=cm.Spectral)
    plt.axis("off")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xf, yf = np.meshgrid(xf, yf, sparse=True)
    ax.plot_surface(xf, yf, ans, cmap=cm.coolwarm,)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

if __name__ == '__main__':
    main()
