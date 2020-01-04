import numpy as np


# Depending on the actual file location
# C:/Users/82569/Desktop/2222.txt
def readfile(location):
    # number of features
    nof = tuple(np.arange(10))

    # ExpFit
    exp = (10,)

    # Extracted matrix
    m = np.loadtxt(location, skiprows=1, usecols=nof)

    # Extracted expFit vector
    v = np.loadtxt(location, skiprows=1, usecols=exp)
    return m, v

loc = 'C:/Users/82569/Desktop/Residuals_Match_DMhydro_Less_z.txt'
m, c = readfile(loc)
