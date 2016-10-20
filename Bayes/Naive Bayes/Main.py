import time

from Utility import *


def main():

    data_time = time.time()
    x, y_data = get_data()
    xy_zip, category, n_possibilities = y_data
    data_time = time.time() - data_time

    learning_time = time.time()
    func = estimate(x, xy_zip, category, discrete_data=n_possibilities)
    learning_time = time.time() - learning_time

    # draw_result(x, y, category, func, draw_border=DRAW_BORDER, show_result=SHOW_RESULT)

    estimation_time = time.time()
    if ESTIMATE_MODEL:
        rs = 0
        categories = [i for i in range(len(category))]
        for xx, yy in xy_zip:
            if predict(xx, func, categories) == yy:
                rs += 1
        print("Acc             : {:12.6} %".format(100 * rs / len(xy_zip)))
    estimation_time = time.time() - estimation_time

    drawing_time = time.time()
    if DRAW_RESULT and len(x[0]) == 2:
        draw_result(x, xy_zip, category, func, SHOW_RESULT)
    drawing_time = time.time() - drawing_time

    if GET_TIME:
        print(
            "Data cleaning   : {:12.6} s\n"
            "Model building  : {:12.6} s\n"
            "Estimation      : {:12.6} s\n"
            "DRAWING         : {:12.6} s\n"
            "Total           : {:12.6} s".format(
                data_time, learning_time, estimation_time, drawing_time,
                data_time + learning_time + estimation_time + drawing_time
            )
        )

main()
