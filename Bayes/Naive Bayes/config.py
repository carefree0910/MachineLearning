SKIP_FIRST = True
TAR_IDX = 16

WHETHER_DISCRETE = [False] + [True] * 15

# If not None, it should be func(category, n_category, dim, n_dim) which returns a list of functions
# Please reference to 'gaussian_maximum_likelihood' in NaiveBayes.py for details
PRE_CONFIGURED_FUNCTION = [None] * 16

MU, SIGMA = [None] * 16, [None] * 16

DRAW_RESULT = True
SHOW_RESULT = True

ESTIMATE_MODEL = True
GET_TIME = True
