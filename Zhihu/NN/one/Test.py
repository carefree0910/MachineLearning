from Models.Zhihu.NN.one.Util import *
from Models.Zhihu.NN.one.Network import *

np.random.seed(142857)  # for reproducibility


def main():

    nn = NNDist()
    epoch = 1000

    timing = Timing(enabled=True)
    timing_level = 1
    nn.feed_timing(timing)

    x, y = DataUtil.gen_xor(10 ** 2, 1)

    # nn.build([x.shape[1], 24, 24, y.shape[1]])
    nn.add("ReLU", (x.shape[1], 24))
    nn.add("CrossEntropy")

    nn.fit(x, y, train_only=True, epoch=epoch, record_period=epoch)
    print("Acc: {:8.6}".format(nn.evaluate(x, y)[0]))

    timing.show_timing_log(timing_level)

if __name__ == '__main__':
    main()
