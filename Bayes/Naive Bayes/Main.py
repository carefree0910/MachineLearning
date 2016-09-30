from Utility import *


def get_data():
    x = []
    y = {}
    category = {}
    with open("data.txt", "r") as file:
        for line in file:
            x.append(list(map(lambda z: float(z), line.split())))
            y[tuple(x[len(x) - 1])] = int(x[len(x) - 1].pop())
        for key, value in y.items():
            try:
                category[value].append(key)
            except KeyError:
                category[value] = [key]

    return x, y, category


def main():
    x, y, category = get_data()

    """scale = 10 ** 2
    per = 1 / scale
    pre_configured_possibilities = [[per * i for i in range(-4 * scale, 5 * scale)],
                                    [per * i for i in range(-4 * scale, 5 * scale)]]
    func = estimate(x, y, category, discrete=pre_configured_possibilities)"""

    func = estimate(x, y, category)

    draw_result(x, y, category, func, draw_border=DRAW_BORDER, show_result=SHOW_RESULT)

main()
