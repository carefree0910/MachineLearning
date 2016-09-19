def dot(x, y):
    rs = 0
    for i in range(len(x)):
        rs += x[i] * y[i]
    return rs


def cost_function(i, x, y, a, b, gram):
    rs = b
    for j in range(len(x)):
        rs += a[j] * y[j] * gram[j][i]
    rs *= y[i]
    return -rs


def gradient_decent(x, y, gram, a=None, b=0, lr=0.1, ceiling=100):
    length = len(x)
    dimension = len(x[0])
    if a is None:
        a = [0 for _ in range(length)]
    flag = 1
    count = 0
    while flag:
        flag = 0
        count += 1
        for i in range(length):
            if cost_function(i, x, y, a, b, gram) >= 0:
                flag = 1
                a[i] += lr
                b += lr * y[i]
        if count >= ceiling:
            break

    return a, b, dimension, flag


def transform(a, x, y, dimension):
    w = [0 for _ in range(dimension)]
    for i in range(len(a)):
        for j in range(dimension):
            w[j] += a[i] * y[i] * x[i][j]
    return w


def predict(i):
    if i > 0:
        return 1
    return -1
