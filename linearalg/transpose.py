import sys
import numpy as np

# Usage: python transpose.py C.csv > C_T.csv
C = np.loadtxt(sys.argv[1], delimiter=",")
C_T = C.T   # transpose

np.savetxt(sys.stdout, C_T, delimiter=",", fmt="%.2f")

