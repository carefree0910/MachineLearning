from NN.Basic.Networks import *
from Util.Util import DataUtil


def main():
    nn = NNDist()
    save = False
    load = False

    lr = 0.001
    lb = 0.001
    epoch = 1000

    timing = Timing(enabled=True)
    timing_level = 1

    x, y = DataUtil.gen_spiral(50, 3, 3, 2.5)

    if not load:
        nn.build([x.shape[1], 6, 6, 6, y.shape[1]])
        nn.optimizer = "Adam"
        nn.preview()
        nn.feed_timing(timing)
        nn.fit(x, y, lr=lr, lb=lb, verbose=1, record_period=4,
               epoch=epoch, batch_size=128, train_only=True,
               draw_detailed_network=True, make_mp4=True, show_animation=True)
        if save:
            nn.save()
        nn.draw_results()
        nn.visualize2d()
    else:
        nn.load()
        nn.preview()
        nn.evaluate(x, y)

    timing.show_timing_log(timing_level)

if __name__ == '__main__':
    main()
