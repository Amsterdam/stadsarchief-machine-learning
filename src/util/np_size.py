import numpy as np


def display_MB(array: np.ndarray):
    print(f'{round(array.nbytes / 1024**2, 3)}MB of type {array.dtype}')
