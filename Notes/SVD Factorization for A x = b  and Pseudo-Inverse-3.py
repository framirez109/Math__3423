# Compute The Singular Value Decomposition of (M x N) Matrix A
# The Pseudo-Inverse of A
# Solve Ax = b using The SVD/Pseudo-Inverse
#
import numpy as np
from scipy.linalg import diagsvd
from scipy.linalg import pinv
#from scipy.linalg import inv
from scipy.linalg import svd
#
# Enter m-Rows & n-Columns Of Coefficient Matrix A:
print(" START ------------------> ")
m = 4 ; n = 3
#
# Generate a Random (m x n) Matrix  A  With Each Integer Entry < 6:
A = np.random.randint(-2, 4, size=(m, n) )
#
print("The (", m, "x", n, ") Random Matrix A =") ; print(A)
#
# Generate a Random (m x 1) Vector b With Each Integer Entry < 6:
b = np.random.randint(5, size=(m, 1) )
#
print("The (", m, "x", 1, ") Random Vector b =") ; print(b)
print(" ------------------ ")
#
#
# For Testing Perposes Set The Solution Vector x:
#x = array([1, 1, 1]) ; print("PreSet x =", x )
#
# And Reset b = A @ x:
#b = A @ x ; print("And Reset b =", b)
#
#
print(" ")
print(" ------------------ ")
print("----> Perform Singular Value Decomposition")
print(" ------------------ ")
U, sigma, VT = svd(A)
print(" The ( ", m, "x", m, ") U Matrix")
print(U)
print(" ------------------ ")
# create m x n Sigma matrix
Sigma = diagsvd(sigma, m, n)
print(" The ( ", m, "x", n, ") Singular Values of A")
print(Sigma)
print(" ------------------ ")
print(" The ( ", n, "x", n, ") V^T Matrix")
print(VT)
print(" ------------------ ")
print("Reconstructed Original","(", m, "x", n, ")","Matrix A From SVD Factors")
print(U @ Sigma @ VT)
print(" ------------------ ")
#
#The Pseudo-Inverse of A is Given by: A^+ = V Sigma^+ U^T
#
print("Compute The ( ", n, "x", m, ") Sigma^+ Singular Values of The Pseudo-Inverse of A")
print(pinv(Sigma))
print(" ------------------ ")
print("---Verify  Sigma * Sigma^+ = I; If A Has Independent Rows ----"  )
print( Sigma @ pinv(Sigma) )
print(" ------------------ ")
print("---Verify  Sigma^+ * Sigma = I; If A Has Independent Columns ----"  )
print( pinv(Sigma) @ Sigma )
print(" ------------------ ")
print(" ---> Compute The Pseudo-Inverse of A <-- (VT).T @ pinv(Sigma) @ U.T")
PseudoA = (VT).T @ pinv(Sigma) @ U.T 
print(PseudoA)
print(" ------------------ ")
print(" ---> Compute The Pseudo-Inverse of A <-- pinv(A)")
print( pinv(A))
print(" ------------------ ")
print(" ---Verify  A * A^+ = I; If A Has Independent Rows    ----" )
print(  A @ PseudoA )
print("---Verify  A^+ * A = I; If A Has Independent Columns   ----" )
print(  PseudoA @ A  )
print(" ------------------ ")
print(" ---Verify  A * A^+ * A = A    ----" )
print(  A @ PseudoA @ A )
print(" ---Verify  A^+ * A * A^+ = A^+    ----" )
print(  PseudoA @ A @ PseudoA )
print(" ------------------ ")
print(" ---> Solve A x = b Using The Pseudo-Inverse of A")
print(" ---> From  x = A^+ b = V Sigma^+ U^T b")
x = PseudoA @  b
#x =  pinv(A) @ b
print(" ------------------ ")
print("Solution Vector x =")
print(x)
print(" ------------------ ")
print("Compute The Residual   A x  - b ")
print(A @ x -b) 
print(" <------------------ END ")
