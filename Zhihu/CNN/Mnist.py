from Util.Util import DataUtil
from Zhihu.CNN.Network import *


def main():
    nn = NNDist()
    verbose = 2

    lr = 0.001
    epoch = 50
    record_period = 5

    timing = Timing(enabled=True)
    timing_level = 1
    nn.feed_timing(timing)

    x, y = DataUtil.get_dataset("mnist", "../../_Data/mnist.txt", quantized=True, one_hot=True)

    nn.add(ReLU((x.shape[1], 400)))
    nn.add(CrossEntropy((y.shape[1],)))

    nn.fit(x, y, lr=lr, epoch=epoch, record_period=record_period, verbose=verbose, train_rate=0.8)
    nn.draw_logs()

    timing.show_timing_log(timing_level)

if __name__ == '__main__':
    main()
