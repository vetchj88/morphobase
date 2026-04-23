import numpy as np

def lesion_cells(hidden: np.ndarray, start: int, stop: int) -> np.ndarray:
    damaged = hidden.copy(); damaged[start:stop] = 0.0; return damaged
