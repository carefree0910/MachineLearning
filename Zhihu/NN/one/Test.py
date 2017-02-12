from Zhihu.NN.one.Network import *
from Zhihu.NN.Layers import *

from Util.Util import DataUtil

np.random.seed(142857)  # for reproducibility


def main():

    nn = NNDist()
    epoch = 1000

    x, y = DataUtil.gen_xor(100)

    nn.add(ReLU((x.shape[1], 24)))
    nn.add(CrossEntropy((y.shape[1],)))

    nn.fit(x, y, epoch=epoch)
    nn.evaluate(x, y)

if __name__ == '__main__':
    main()
