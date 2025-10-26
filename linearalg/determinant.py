import sys
import numpy as np

# Usage: python determinant.py C_T.csv > det.txt
C_T = np.loadtxt(sys.argv[1], delimiter=",")
det = np.linalg.det(C_T)   # determinant

print(f"Determinant: {det:.4f}")

