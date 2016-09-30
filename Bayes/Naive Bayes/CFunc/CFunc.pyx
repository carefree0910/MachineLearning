# ======================================================================================================================
# Extern

cdef extern from "math.h":
    double exp(double)

cdef extern from "math.h":
    double sqrt(double)

cdef extern from "math.h":
    double log(double)


# ======================================================================================================================
# Functions

def gaussian(double x, double mu, double sigma):
    return exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sqrt(6.2832) * sigma)
