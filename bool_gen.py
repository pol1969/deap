import numpy as np

def binary_mask_random(r, c, n):
    a = np.zeros((r,c)).flatten()

    for i in range(np.random.randint(0, n+1)):
        x = np.random.randint(0, r*c)
        a[x] = 1

    return a.reshape((r,c))
