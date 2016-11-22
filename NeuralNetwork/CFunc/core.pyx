# encoding: utf8

import numpy as np
cimport numpy as np

ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t

def bp_core(
        int N, int C, int F, int H, int W, int OH, int OW, int HH, int WW, int p, int sd,
        np.ndarray[DTYPE_t, ndim=4] inner_weight,
        np.ndarray[DTYPE_t, ndim=4] delta):
    cdef np.ndarray[DTYPE_t, ndim=4] x_padded = np.zeros((N, C, H + 2 * p, W + 2 * p), dtype=delta.dtype)

    cdef int i, f, j, k
    for i in range(N):
        for f in range(F):
            for j in range(OH):
                for k in range(OW):
                    x_padded[i, :, j*sd:HH+j*sd, k*sd:WW+k*sd] += (inner_weight[f] * delta[i, f, j, k])

    if p > 0:
        return x_padded[:, :, p:-p, p:-p]
    return x_padded
