import sys
import numpy as np

# Usage: python multiply.py A.csv B.csv > C.csv
A = np.loadtxt(sys.argv[1], delimiter=",")
B = np.loadtxt(sys.argv[2], delimiter=",")
C = A @ B   # matrix multiplication

np.savetxt(sys.stdout, C, delimiter=",", fmt="%.2f")
