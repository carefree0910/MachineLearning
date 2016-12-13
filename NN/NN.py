try:
    from TF.Networks import *
    print("Using tensorflow backend")
except ImportError:
    from Basic.Networks import *
    print("Using numpy backend")
