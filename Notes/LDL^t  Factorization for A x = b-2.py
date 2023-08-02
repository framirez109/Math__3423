# Given A^t = A Solve  A x = b Using The LDL^t Factorization 
#
from numpy import array
from scipy.linalg import ldl
from scipy.linalg import solve, det
#
print(" ")
print("Original (m x m) Symmetric Matrix A = A^t  (m x 1) Vector b")
#
A  = array([[   13,  0,     0],
            [   10,  2,     0],
            [  100,  0,    -4]], float)
#
A  = array([[ 7, 3,  2, 0],
            [ 3, 6,  1, 1],
            [ 2, 1, 9,  3],
            [ 0, 1, 3, 7]], float)

m = 4 
#
b = array([2, 2, 1, 2])
#
print("----- The (m x m) Matrix A -----")
print(A)
print(" ")
print("----- The (m x 1) Vector b  -----")
print(b)
print(" ")
#
print("----- Factorization of A to LDL^t -----")
#
L, D, P = ldl(A)
#
print(" ")
print("----- Triangular (m x m) Matrix L ----- ")
print(L[P, :])
print(" ")
print("----- Diagonal (m x m) Matrix  D ----- ")
print(D)
print(" ")
print("----- Reconstruct A = L D L^t ----- ")
print(L @ D @ L.T)
print(" ")
print(" -----  Compute Determinant of A =", det(D))
print(" ")
print(" -----  Solve A x = L D L^t x = b  ----- ")
print("  ")
print(" -----  1st: Solve L z = b  for z ----- ")
z = solve(L, b)
print(" ----- The (m x 1) Vector z =", z)
print(" ")
print(" ----- 2nd: Solve D y = z  for y ----- ")
y = solve(D, z)
print(" ----- The (m x 1) Vector y =", y)
print(" ")
print(" ----- 3rd: Solve L^t x = y  for x ----- ")
x = solve(L.T, y)
print(" ----- The (m x 1) Solution Vector x =", x)
print(" ")
print(" ----- Reconstruct the (m x 1) b Vector")
print(" ----- Is  L @ D @ L.T @ x = b =",  L @ D @ L.T @ x )
print(" ")