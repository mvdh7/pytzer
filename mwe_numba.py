from numba import jit
import numpy as np
import time

# jit decorator tells Numba to compile this function.
# The argument types will be inferred by Numba when function is called.
def sum2d_normal(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result

@jit
def sum2d(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result

a = np.arange(12).reshape(4,3)
go = time.time()
print(sum2d(a))
print(time.time() - go)

go = time.time()
print(sum2d_normal(a))
print(time.time() - go)
