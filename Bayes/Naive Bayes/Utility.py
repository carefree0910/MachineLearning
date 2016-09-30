import matplotlib.pyplot as plt
import matplotlib.cm as cm

from NaiveBayes import *


def draw_result(x, y, category, func, draw_border=False, not_continuous=None, show_result=True):
    categories = [key for key in category]
    axis = [[xx[i] for xx in x] for i in range(len(x[0]))]
    per = 1 / len(categories)
    colors = cm.rainbow([i * per for i in range(len(categories))])

    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    ax1.set_title("Original")
    ax2.set_title("Prediction")
    ax1.scatter(axis[0], axis[1], color=[colors[y[tuple(xx)]] for xx in x])
    ax2.scatter(axis[0], axis[1], color=[colors[predict(xx, func, categories)] for xx in x])

    if draw_border:
        if not_continuous is None:
            bx = []
            by = []

            i, j = int(min(axis[0])) - 2, int(min(axis[1])) - 2
            x_ceiling, y_ceiling = int(max(axis[0])) + 2, int(max(axis[1])) + 2

            plt.ylim(j, y_ceiling)
            plt.xlim(i, x_ceiling)

            while i <= int(max(axis[0])) + 2:
                while j <= int(max(axis[1])) + 2:
                    if draw_border_core_process(i, j, func, categories):
                        bx.append(i)
                        by.append(j)
                    j += GAP
                i += GAP
                j = int(min(axis[1])) - 2
        else:
            bx = []
            by = []
            for i in not_continuous[0]:
                for j in not_continuous[1]:
                    if draw_border_core_process(i, j, func, categories):
                        bx.append(i)
                        by.append(j)

        ax2.scatter(bx, by, s=BORDER_SCALE, color="black")

    if show_result:
        plt.show()
