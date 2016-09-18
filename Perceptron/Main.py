import matplotlib.pyplot as plt

from Perceptron import *


def main(read_data, lr=0.1, num=50, dimension=2, scale=5, period=2, repeat=20):

    x = []
    y = []

    if read_data:
        with open("data.txt", "r") as file:
            for line in file:
                x.append(list(map(lambda z: float(z), line.split())))
                y.append(x[len(x) - 1].pop())
    else:
        for i in range(num):
            x.append([random() * scale for _ in range(dimension)])
            y.append(-1)
            x.append([random() * scale + scale for _ in range(dimension)])
            y.append(1)

    rs = "failed"
    w = b = 0
    for i in range(repeat):
        if not i:
            w, b, dimension, flag = gradient_decent(x, y, lr=lr, ceiling=period)
        else:
            w, b, dimension, flag = gradient_decent(x, y, w=w, b=b, lr=lr, ceiling=period)

        if dimension == 2:
            w1, w2 = w
            xs = [i for i in range(0, scale * 2)]
            axis = [[x[i][j] for i in range(len(x))] for j in range(len(x[0]))]

            plt.figure()
            plt.scatter(axis[0], axis[1], c=[(0, 0, 0) if y[i] > 0 else (1, 1, 1) for i in range(len(y))])
            plt.plot(xs, [(-b - w1 * xs[i]) / w2 for i in range(len(xs))])
            plt.show()

        if not flag:
            rs = "success"
            break

    print(rs)
    print("w: {}; b: {}".format(w, b))
    for i in range(len(x)):
        print("org: {}; predict: {}".format(y[i], dot(w, x[i]) + b))

main(read_data=True, period=200, repeat=50, lr=0.05)
