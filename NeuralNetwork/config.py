# ==============================
# Basic Config

TAR_IDX = 2
CLEAR_CACHE = True
SKIP_FIRST = False

SHOW_FIGURE = True


# ==============================
# Neural Network

# BETA measures importance of positive recall
BETA = 1
beta_2 = BETA ** 2

# Boost Positive Samples
BOOST_LESS_SAMPLES = False

# Main Params
DATA_CLEANED = True
TRAIN_ONLY = True

CLASSES_NUM = 2
WHETHER_NUMERICAL = [True] * 16 + [False] * 3
WHETHER_EXPAND = [False] * 9 + [True] * 7 + [False] * 4
EXPAND_NUM_LST = [0] * 9 + [
    12, 4, 8, 2, 10, 5, 3
] + [0] * 4

TRAINING_SCALE = 0.7
CV_SCALE = 0.15

EPOCH = 40000
BATCH_SIZE = 6
LEARNING_RATE = 10 ** -3

RECORD_PERIOD = 10000
