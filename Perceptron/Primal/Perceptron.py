from random import random


def dot(x, y):
    rs = 0
    for i in range(len(x)):
        rs += x[i] * y[i]
    return rs


def cost_function(xi, yi, w, b):
    return -(dot(w, xi) + b) * yi


def gradient_decent(x, y, w=None, b=0, lr=0.1, ceiling=100):
    length = len(x)
    dimension = len(x[0])
    if w is None:
        w = [random() for _ in range(dimension)]
    flag = 1
    count = 0
    while flag:
        flag = 0
        count += 1
        for i in range(length):
            if cost_function(x[i], y[i], w, b) >= 0:
                flag = 1
                for j in range(dimension):
                    w[j] += lr * x[i][j] * y[i]
                b += lr * y[i]
        if count >= ceiling:
            break
    return w, b, dimension, flag


def predict(i):
    if i > 0:
        return 1
    return -1
