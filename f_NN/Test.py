from Util.Util import DataUtil

backend = "Basic"

if backend == "Basic":
    from f_NN.Basic.Networks import *
    from f_NN.Basic.Layers import *
else:
    from f_NN.TF.Networks import *
    from f_NN.TF.Layers import *


def main():

    nn = NN()
    epoch = 1000

    x, y = DataUtil.gen_spin(50, 12, 2, 4)

    nn.add(ReLU((x.shape[1], 24)))
    nn.add(ReLU((24,)))
    if backend == "Basic":
        nn.add(Softmax((y.shape[1],)))
    else:
        nn.add(CrossEntropy((y.shape[1],)))

    nn.fit(x, y, epoch=epoch)
    nn.estimate(x, y)
    nn.visualize2d(x, y)
    nn.show_timing_log()

if __name__ == '__main__':
    main()
