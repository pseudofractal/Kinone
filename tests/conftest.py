import os
import random
import numpy as np

SEED = 42

def pytest_configure():
    random.seed(SEED)
    np.random.seed(SEED)

