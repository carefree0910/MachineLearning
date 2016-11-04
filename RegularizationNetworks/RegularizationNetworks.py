import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sigma = 1
lb = 0.1


def gaussian_kernel(x):
    return np.exp(-np.sum(np.array(x) ** 2, axis=2) * 0.5 / sigma ** 2)


def gen_spin(size=100, n_classes=3):
    dimension = 2
    xs = np.zeros((size * n_classes, dimension))
    ys = np.zeros((size * n_classes, n_classes), dtype='uint8')
    for j in range(n_classes):
        ix = range(size * j, size * (j + 1))
        r = np.linspace(0.0, 1, size)
        t = np.array(
            np.linspace(j * (n_classes + 1), (j + 1) * (n_classes + 1), size) +
            np.array(np.random.random(size=size)) * 0.2)
        xs[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        ys[ix, j] = 1

    return np.array([xs]), ys


def main():

    x, y = gen_spin()

    x_matrix = (x - x.reshape((x.shape[1], 1, 2))).reshape((x.shape[1], x.shape[1], 2))
    c = np.linalg.solve(gaussian_kernel(x_matrix) + lb * np.array(np.eye(len(y))), y)

    def f(_x):
        return np.dot(gaussian_kernel(_x - x), c)

    plot_scale = 2
    plot_num = 100
    x_min, x_max = min(-1, np.min(x)), max(1, np.max(x))

    xf = np.linspace(x_min * plot_scale, x_max * plot_scale, plot_num)
    yf = np.linspace(x_min * plot_scale, x_max * plot_scale, plot_num)
    x_base, y_base = np.meshgrid(xf, yf)
    base_matrix = np.dstack((x_base, y_base)).reshape((plot_num * plot_num, 1, 2))
    ans = f(base_matrix).reshape((plot_num, plot_num, c.shape[1]))

    plt.contourf(x_base, y_base, np.argmax(ans, axis=2), cmap=cm.Spectral)
    plt.scatter(x[0][:, 0], x[0][:, 1], c=np.argmax(y, axis=1), s=40, cmap=cm.Spectral)
    plt.axis("off")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xf, yf = np.meshgrid(xf, yf, sparse=True)
    ax.plot_surface(xf, yf, np.max(ans, axis=2), cmap=cm.coolwarm,)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

if __name__ == '__main__':
    main()
