import numpy as np

def local_prediction(hidden: np.ndarray) -> np.ndarray:
    return np.tanh(hidden)
