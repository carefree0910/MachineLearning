import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def do_visualization(x, y, func, plot_scale, plot_num):

    x_min, x_max = min(-1, np.min(x)), max(1, np.max(x))

    xf = np.linspace(x_min * plot_scale, x_max * plot_scale, plot_num)
    yf = np.linspace(x_min * plot_scale, x_max * plot_scale, plot_num)
    x_base, y_base = np.meshgrid(xf, yf)
    base_matrix = np.dstack((x_base, y_base)).reshape((plot_num * plot_num, 1, 2))
    ans = func(base_matrix).reshape((plot_num, plot_num, y.shape[1]))

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
