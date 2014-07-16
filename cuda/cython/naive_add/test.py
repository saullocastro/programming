import cProfile
import numpy as np

import my_cuda

n = 15000000
a = np.random.random(n).astype(np.float32)
b = np.random.random(n).astype(np.float32)

sum1 = my_cuda.add(a, b, parallel='block')
sum2 = my_cuda.add(a, b, parallel='thread')
sum3 = my_cuda.add(a, b, parallel='block-thread')
sum4 = my_cuda.add(a, b, parallel='long-array')
check = np.allclose(sum1, a+b)
print('Sum using parallel block is correct: {0}'.format(check))
check = np.allclose(sum2, a+b)
print('Sum using parallel thread is correct: {0}'.format(check))
check = np.allclose(sum3, a+b)
print('Sum using parallel block-thread is correct: {0}'.format(check))
check = np.allclose(sum4, a+b)
print('Sum using parallel long-array is correct: {0}'.format(check))
