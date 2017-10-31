import os
import sys
root_path = os.path.abspath("../")
if root_path not in sys.path:
    sys.path.append(root_path)

from f_NN.Networks import *
from f_NN.Layers import *

from Util.Util import DataUtil


def main():

    nn = NN()
    epoch = 1000

    x, y = DataUtil.gen_spiral(120, 7, 7, 4)

    nn.add(ReLU((x.shape[1], 24)))
    nn.add(ReLU((24, )))
    nn.add(CostLayer((y.shape[1],), "CrossEntropy"))

    # nn.disable_timing()
    nn.fit(x, y, epoch=epoch, train_rate=0.8, metrics=["acc"])
    nn.evaluate(x, y)
    nn.visualize2d(x, y)
    nn.show_timing_log()
    nn.draw_logs()

if __name__ == '__main__':
    main()
