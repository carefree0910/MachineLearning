import matplotlib.pyplot as plt
import matplotlib.cm as cm

from NaiveBayes import *


def data_cleaning(line):
    line = line.replace('"', "")
    return list(map(lambda c: c.strip(), line.split(";")))


def get_data():
    categories = None
    line_len = 0

    x = []
    with open("Data/data.txt", "r") as file:
        flag = None
        for line in file:
            if SKIP_FIRST and flag is None:
                flag = True
                continue

            line = data_cleaning(line)

            tmp_x = []
            if categories is None:
                line_len = len(line)
                categories = [{ "flag": 1, line[i]: 0 } for i in range(line_len)]
            for i in range(line_len):
                if line[i] in categories[i]:
                    tmp_x.append(categories[i][line[i]])
                else:
                    tmp_x.append(categories[i]["flag"])
                    categories[i][line[i]] = categories[i]["flag"]
                    categories[i]["flag"] += 1

            x.append(tmp_x)

    y = []
    for xx in x:
        y.append(xx.pop(TAR_IDX))
    xy_zip = list(zip(x, y))
    category = [[] for _ in range(max(y) + 1)]

    for xx, yy in xy_zip:
        category[yy].append(xx)

    n_possibilities = [categories[i]["flag"] if WHETHER_DISCRETE[i] else PRE_CONFIGURED_FUNCTION[i]
                       for i in range(line_len) if i != TAR_IDX]
    y_data = (xy_zip, category, n_possibilities)

    return x, y_data


def draw_result(x, xy_zip, category, func, show_result=True):
    categories = [key for key in category]
    axis = [[xx[i] for xx in x] for i in range(len(x[0]))]
    per = 1 / len(categories)
    colors = cm.rainbow([i * per for i in range(len(categories))])

    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    ax1.set_title("Original")
    ax2.set_title("Prediction")
    ax1.scatter(axis[0], axis[1], color=[colors[_y] for _, _y in xy_zip])
    ax2.scatter(axis[0], axis[1], color=[colors[predict(xx, func, categories)] for xx in x])

    if show_result:
        plt.show()
