Naive addition of two vectors
=============================

This implements a function that adds two arrays just to show how to import a CUDA module in Cython.

Installing
----------

One should run::

    nvcc --use-local-env --cl-version 2010 -lib -o cuda_blockadd.lib cuda_blockadd.cu
    nvcc --use-local-env --cl-version 2010 -lib -o cuda_threadadd.lib cuda_threadadd.cu
    nvcc --use-local-env --cl-version 2010 -lib -o cuda_btadd.lib cuda_btadd.cu
    nvcc --use-local-env --cl-version 2010 -lib -o cuda_longadd.lib cuda_longadd.cu

And then::

    python setup.py build_ext -i -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.0\include" -lcudart -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.0\lib\x64" --force clean

.. note::

    Some parameters may change according to your system!

.. note::

    The compiling steps are included in ``compile.bat`` so one can just run
    it.

Testing
-------

Thre testing routine will sum two random 1-D arrays with ``n=15000000`` terms
and compare the results with ``numpy``. Only the ``parallel='block-thread'``
should give the correct result::

   python test.py

Which will produce something like::

    Sum using parallel block is correct: False
    Sum using parallel thread is correct: False
    Sum using parallel block-thread is correct: True
    Sum using parallel long-array is correct: True

Timing
------

Timing the performance of this naive add function in IPython::

    timeit np.add(a, b)
    10 loops, best of 3: 26.3 ms per loop

    timeit my_cuda.add(a, b, parallel='block-thread')
    10 loops, best of 3: 108 ms per loop

    timeit my_cuda.add(a, b, parallel='long-array')
    10 loops, best of 3: 108 ms per loop

which is already pretty good...

Comments
--------

The implementation using ``parallel='block'`` is limited to arrays up to
``n=65535`` which is the maximum number of blocks, and for
``parallel='thread'`` to arrays up to ``n=512`` which is the maximum number of
threads per execution block.

The implementation using ``parallel='block-thread'`` is limited to
``n=65535*256=16776960`` in the way it is implemented and it could be
expanded, but the solution in ``parallel='long-array'`` is easier to manage
and gives the same performance, being unlimited to the size of the array and
limited only by the amount of RAM memory in the GPU.

Possible Improvements
---------------------

Do one cache-friendly access in the ``'long-array'`` implementation
since the pace of ``blockDim.x * gridDim.x`` seems to require a large stride
that will probably result in many cache losses (not sure though...).
