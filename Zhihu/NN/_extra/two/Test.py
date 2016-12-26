from Zhihu.NN.Util import *
from Zhihu.NN._extra.two.Networks import *

np.random.seed(142857)  # for reproducibility


def main():

    nn = NNDist()
    epoch = 1600

    timing = Timing(enabled=True)
    timing_level = 1
    nn.feed_timing(timing)

    x, y = DataUtil.gen_spin(1000)

    nn.add(ReLU((x.shape[1], 36)))
    nn.add(ReLU((36,)))
    nn.add(Softmax((y.shape[1],)))

    nn.fit(x, y, epoch=epoch, verbose=2)
    nn.draw_logs()
    nn.visualize_2d(x, y)

    timing.show_timing_log(timing_level)

if __name__ == '__main__':
    main()
