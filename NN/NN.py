# TODO: Remove dummy Timings

try:
    from NN.TF.Networks import *
    print("Using tensorflow backend")
except ImportError:
    from NN.Basic.Networks import *
    print("Using numpy backend")
