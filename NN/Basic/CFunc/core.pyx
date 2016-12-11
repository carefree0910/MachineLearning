# encoding: utf8

import numpy as np
cimport numpy as np
cimport cython

ctypedef fused DTYPE_t:
    np.float32_t
    np.float64_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef col2im_6d_cython_inner(np.ndarray[DTYPE_t, ndim=6] cols,
                            np.ndarray[DTYPE_t, ndim=4] x_padded,
                            int N, int C, int H, int W, int HH, int WW,
                            int out_h, int out_w, int pad, int stride):

    cdef int c, hh, ww, n, h, w
    for n in range(N):
        for c in range(C):
            for hh in range(HH):
                for ww in range(WW):
                    for h in range(out_h):
                        for w in range(out_w):
                            x_padded[n, c, stride * h + hh, stride * w + ww] += cols[c, hh, ww, n, h, w]



def col2im_6d_cython(np.ndarray[DTYPE_t, ndim=6] cols, int N, int C, int H, int W,
        int HH, int WW, int pad, int stride):
    cdef np.ndarray x = np.empty((N, C, H, W), dtype=cols.dtype)
    cdef int out_h = (H + 2 * pad - HH) / stride + 1
    cdef int out_w = (W + 2 * pad - WW) / stride + 1
    cdef np.ndarray[DTYPE_t, ndim=4] x_padded = np.zeros((N, C, H + 2 * pad, W + 2 * pad),
                                                  dtype=cols.dtype)

    col2im_6d_cython_inner(cols, x_padded, N, C, H, W, HH, WW, out_h, out_w, pad, stride)

    if pad > 0:
        return x_padded[:, :, pad:-pad, pad:-pad]
    return x_padded
