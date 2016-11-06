from Util import *

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

    do_visualization(x, y, f, plot_scale, plot_num)

if __name__ == '__main__':
    main()
