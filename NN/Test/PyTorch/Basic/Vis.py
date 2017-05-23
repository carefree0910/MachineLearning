from NN.PyTorch.Basic.Networks import *

from Util.Util import DataUtil


def main():
    nn = NNDist()
    save = False
    load = False

    lr = 0.001
    lb = 0.001
    epoch = 1000
    record_period = 4

    x, y = DataUtil.gen_spiral(50, 3, 3, 2.5)

    if not load:
        nn.build([x.shape[1], 6, 6, 6, y.shape[1]])
        nn.optimizer = "Adam"
        nn.preview()
        nn.fit(x, y, lr=lr, lb=lb, verbose=1, record_period=record_period,
               epoch=epoch, batch_size=128, train_only=True,
               animation_params={"show": True, "mp4": False, "period": record_period})
        if save:
            nn.save()
        nn.visualize2d(x, y)
        nn.draw_results()
    else:
        nn.load()
        nn.preview()
        nn.evaluate(x, y)

    nn.show_timing_log()

if __name__ == '__main__':
    main()
