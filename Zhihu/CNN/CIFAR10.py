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

    x, y = DataUtil.get_dataset("cifar10", "../../_Data/cifar10.txt", quantized=True, one_hot=True)
    x = x.reshape(len(x), 3, 32, 32)

    nn.add(ConvReLU((x.shape[1:], (32, 3, 3))))
    nn.add(ConvReLU(((32, 3, 3),)))
    nn.add(MaxPool(((3, 3),), 2))
    nn.add(ConvReLU(((64, 3, 3),)))
    nn.add(ConvReLU(((64, 3, 3),)))
    nn.add(AvgPool(((3, 3),), 2))
    nn.add(ConvReLU(((32, 3, 3),)))
    nn.add(ConvReLU(((32, 3, 3),)))
    nn.add(AvgPool(((3, 3),), 2))
    nn.add(ReLU((512,)))
    nn.add(ReLU((64,)))
    nn.add(CrossEntropy((y.shape[1], )))

    nn.fit(x, y, lr=lr, epoch=epoch, record_period=record_period, verbose=verbose, train_rate=0.8)
    nn.draw_logs()

    timing.show_timing_log(timing_level)

if __name__ == '__main__':
    main()
