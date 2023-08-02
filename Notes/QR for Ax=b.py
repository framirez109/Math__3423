# Solution of A x = b Using The QR Factorizations
#
from numpy import array
from scipy.linalg import qr
from scipy.linalg import solve
#
print("Original (m x n) Matrix A  &  (m x 1) Vector b")
#
A = array([[ 0, 0 ],
           [ 2, 1],
           [ 0, 2]])
#
b = array([1, 2, 3])
#
print(A)
print(" ------------------ ")
print("b =", b)
print(" ------------------ ")
print(" QR - Factorization of A")
Q, R = qr(A) 
print(" ------------------ ")
#
print("Matrix Q (m x m)")
print(Q)
print(" ---------- ")
print("Matrix Q^T Q = I (m x m)")
print(Q.T @ Q )
print(" ---------- ")
print("Matrix R (m x n)")
print(R)
print(" ---------- ")
print("Reconstruct A from QR Factorization")
print( Q @ R )
print(" ------------------ ")
print(" ---> Solve A x = b Using the QR Factorization")
print(" ---> From A = Q R; we have A x = Q R x = b ")
print(" ------------------ ")
print(" ---> 1st: Solve Q y = b --> y ")
y = solve(Q, b)
print(" Vector y =", y)
print(" ------------------ ")
print(" ---> 2nd: Solve R^T R x = R^T y --> x ")
x = solve(R.T @ R, R.T @ y)
print("Solution Vector x =", x)
print(" ------------------ ")
print("Verify that A x = b = ",  A @ x) 
print(" -------------> END")