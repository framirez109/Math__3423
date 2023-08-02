# 
# QR Factorization of Rectangular ( m x n ) Matrix A
#
import numpy as np 
#from numpy import array
from scipy.linalg import qr
#from scipy.linalg import inv
#from scipy.linalg import pinv
from scipy.linalg import solve
#
print(" START ------------------> ")
# Enter m-Rows & n-Columns Of Coefficient Matrix A:
m = 5 ; n = 3
# If choice = 0, mode='economic' Q is (m x n) & R is (n x n)
# If choice = 1, mode='full'     Q is (m x m) & R is (m x n)
choice = 0 
#
# Generate a Random (m x n) Matrix  A  With Each Integer Entry < 6:
A = np.random.randint(-2,5 , size=(m, n) )
# 
print("The (", m, "x", n, ") Random Matrix A =") ; print(A)
#
# Generate a Random (m x 1) Vector b With Each Integer Entry < 9:
b = np.random.randint(-2,3, size=(m, 1) )
#
print("The (", m, "x", 1, ") Random Vector b =") ; print(b)
print(" ------------------ ")
#
# QR - Factorization of A
if choice == 0: 
     Q, R = qr(A, mode='economic')
else:
     Q, R = qr(A, mode='full')
#
print("Matrix Q (", m, "x", n, ")  OR  (", m, "x", m, ")")
print(Q)
print(" ---------- ")
print("Matrix Q^T Q = I is (", n, " x ", n, ")  OR  (", m, " x ", m, ")")
print(Q.T @ Q )
print(" ---------- ")
print("Matrix R (", n, "x", n, ")  OR  (", m, "x", n, ")")
print(R)
print(" ---------- ")
print("Reconstruct A from QR")
print( Q @ R )
print(" ")
#
if choice == 0: 
  x = solve(R, Q.T @ b)
  print("---- QR Solution of  A x = Q R x = b -----")
  print("---- From The Solution of R x = Q.T b -----")
  print(" ")
  print("---- Solution Vector x =  -----")
  print(x)
  print(" ------------------ ")
  print("Compute Residual = A x - b ") 
  print( A @ x - b)
  print(" ")
#
#################################################################
#
print("----- QR Solution of A.T A x = R.T R x = R.T Q.T b = A.T b -----")
print("----- From The Solution of: R.T R x = A.T b -----")
print(" ")
print("----- Solution Vector x  of: R.T R x = A.T b  -----")
x = solve(R.T @ R, A.T @ b)
print(x)
print(" ")
#
print("----- Solution Vector x of:  A.T A x = A.T b  -----")
x = solve(A.T @ A, A.T @ b)
print(x)
print(" ")
#
print("----- Compute Residual = A.T A x - A.T b  -----") 
print( A.T @ A @ x - A.T @ b) 
print(" <------------------ END ")
