from Zhihu.NN.Util import *
from Zhihu.NN.one.Network import *
from Zhihu.NN.Layers import *

np.random.seed(142857)  # for reproducibility


def main():

    nn = NNDist()
    epoch = 1600

    timing = Timing(enabled=True)
    timing_level = 1
    nn.feed_timing(timing)

    x, y = DataUtil.gen_xor(100)

    nn.add(ReLU((x.shape[1], 24)))
    nn.add(CrossEntropy((y.shape[1],)))

    nn.fit(x, y, epoch=epoch)
    nn.evaluate(x, y)

    timing.show_timing_log(timing_level)

if __name__ == '__main__':
    main()
