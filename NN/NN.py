try:
    from Tensorflow.Networks import *
except ImportError:
    print("Tensorflow backend is not available. A pure numpy-implemented backend will be used.")
    from Basic.Networks import *
