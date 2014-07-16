import numpy as np
cimport numpy as np

cdef extern from './cuda_blockadd.h':
    int blockmain(float *a, float *b, float *out, int n) nogil
cdef extern from './cuda_threadadd.h':
    int threadmain(float *a, float *b, float *out, int n) nogil
cdef extern from './cuda_btadd.h':
    int btmain(float *a, float *b, float *out, int n) nogil
cdef extern from './cuda_longadd.h':
    int longmain(float *a, float *b, float *out, int n) nogil

def add(np.ndarray[np.float32_t, ndim=1] a,
        np.ndarray[np.float32_t, ndim=1] b, parallel='block'):
    cdef np.ndarray[np.float32_t, ndim=1] out
    n = a.shape[0]
    assert b.shape[0] == a.shape[0]
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    out = np.empty_like(a)
    if parallel=='block':
        blockmain(&a[0], &b[0], &out[0], n)
    elif parallel=='thread':
        threadmain(&a[0], &b[0], &out[0], n)
    elif parallel=='block-thread':
        btmain(&a[0], &b[0], &out[0], n)
    elif parallel=='long-array':
        longmain(&a[0], &b[0], &out[0], n)
    else:
        raise ValueError('{0} is an invalid parallel possibility!'.format(
                         parallel))

    return out
