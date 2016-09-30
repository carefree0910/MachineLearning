MULTIVARIATE_NORMAL = False
USE_SCIPY_NORM = False

if MULTIVARIATE_NORMAL:
    pre_configured_sigma = {
        0: [
            [1, 0],
            [0, 1]
        ],
        1: [
            [1, 0],
            [0, 1]
        ],
        2: [
            [1, 0],
            [0, 1]
        ],
    }
else:
    pre_configured_sigma = {
        0: [1, 1],
        1: [1, 1],
        2: [1, 1],
    }

MU = None
SIGMA = None

GAP = 5 * 10 ** -3
EPSILON = 10 ** -5

DRAW_BORDER = True
SHOW_RESULT = True

BORDER_SCALE = 2
