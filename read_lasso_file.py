import numpy as np


# Depending on the actual file location
def readfile(location):
    # number of features
    nof = tuple(np.arange(8))

    # ExpFit
    exp = (8,)

    # Extracted matrix
    m = np.loadtxt(location, skiprows=1, usecols=nof)

    # Extracted expFit vector
    v = np.loadtxt(location, skiprows=1, usecols=exp)
    return m, v



