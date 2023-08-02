# Solution of A^T A x = b Using Solve
# Solution of A^T A x = b Using The Inverse
# Solution of A^T A x = b Using The PLU Factorization
# Solution of A^T A x = b Using The LDL^t Factorization
# Solution of A^T A x = b Using The QR Factorization 
# Solution of A^T A x = b Using The SVD Factorization
# Solution of A^T A x = b Using The Pseudo-Inverse 
# 
from numpy import array
from scipy.linalg import inv
from scipy.linalg import solve
from scipy.linalg import lu 
from scipy.linalg import ldl
from scipy.linalg import qr
from scipy.linalg import diagsvd
from scipy.linalg import svd
from scipy.linalg import pinv
#
print("Original (m x n) Matrix A  &  (m x 1) Vector b")
#
A  = array([[1, 2, 5], 
            [4, 5, 6]])
m = 2
n = 3
#
b = array([1, 2, 3])
#
ATA = A.T @ A
#
print("----- The (n x n) Matrix ATA -----------------------")
print(ATA)
print(" ")
print("----- The (n x 1) Matrix b  -----------------------")
print(b)
print(" ")
################################################################
print(" Solution of The A^t A x = b Using solve ")
#
x = solve(ATA,b)
# 
print(" ")
print(" ----- The (n x 1) Solution Vector x -----")
print(x)
print(" ")
print(" ----- Reconstruct (n x 1) b Vector")
b = ATA @ x
print(b)
print(" ")
################################################################
print(" Solution of The A^t A x = b Using The Inverse ")
#
INV_ATA = inv(ATA)
x = INV_ATA @ b
#
print(" ")
print(" ----- The (n x n) Inverse of A^t A -----")
print(ATA @ INV_ATA)
print(" ")
print(" ----- The (n x 1) Solution Vector x -----")
print(x)
print(" ")
print(" ----- Reconstruct (n x 1) b Vector")
b = ATA @ x
print(b)
print(" ")
#################################################################
print(" Solution of A^T A x = b Using The PLU Factorization ")
#
P, L, U = lu(ATA) 
#
print(" ")
print("----- Permutation (n x n) P Matrix ----- ")
print(P)
print(" ")
print("----- Lower Triangular (n x n) L Matrix ----- ")
print(L)
print(" ")
print("----- Upper Triangular (n x n) U Matrix ----- ")
print(U)
print(" ")
print("----- Reconstruct ATA = P L U ----- ")
ATA = P @ L @ U
print(ATA)
print(" ")
#
print(" -----  Solve A^t A x = b Using the PLU-Factorization ----- ")
print(" -----  Since A^t A = PLU we have L U x = P^t b ----- ")
print("  ") 
print(" -----  1st: Solve L y = P^t b  for y ----- ")
y = solve(L, P.T @ b)
print(" ----- The (n x 1) Vector y -----")
print(y)
print(" ")
print(" ----- 2nd: Solve U x = y  for x ----- ")
x = solve(U, y)
print(" ----- The (n x 1) Solution Vector x -----")
print(x)
print(" ")
print(" ----- Reconstruct (n x 1) b Vector")
b = P @ L @ U @ x
print(b)
print(" ")
#########################################################
print(" Solution of A^T A x = b Using LDL^t - Factorization ")
#
L, D, p = ldl(ATA)
#
print(" ")
P = diagsvd(p, n, n)
print("----- Permutation (n x n) P Matrix ----- ")
print(P)
print(" ")
print("----- Lower Triangular (n x n) L Matrix ----- ")
print(L)
print(" ")
print("----- Diagonal (n x n) D Matrix ----- ")
print(D)
print(" ")
print("----- Reconstruct ATA = L D L^t ----- ")
ATA = L @ D @ L.T
print(ATA)
print(" ")
print(" -----  Solve A^t A x = b Using the LDL^t - Factorization ----- ")
print(" -----  Since A^t A = PLDL^t we have L D L^t x = P^t b ----- ")
print("  ")
print(" -----  1st: Solve L z = P^t b  for z ----- ")
z = solve(L, b)
print(" ----- The (n x 1) Vector z -----")
print(z)
print(" ")
print(" ----- 2nd: Solve D y = z  for y ----- ")
y = solve(D, z)
print(" ----- The (n x 1) Solution Vector y -----")
print(y)
print(" ")
print(" ----- 3rd: Solve L^t x = y  for x ----- ")
x = solve(L.T, y)
print(" ----- The (n x 1) Solution Vector x -----")
print(x)
print(" ")
print(" ----- Reconstruct (n x 1) b Vector")
b = P @ L @ L.T @ x
print(b)
print(" ")
##########################################################
print(" Solution Using QR  - Factorization of the (n x n) A^T A")
Q, R = qr(ATA)
print("  ")
#
print("Matrix Q (m x m)")
print(Q)
print("  ")
print("Matrix Q^T Q = I (m x m)")
print(Q.T @ Q )
print("  ")
print("Matrix R (m x m)")
print(R)
print("  ")
print("Confirm A^T A - Q R = 0")
print( Q @ R - ATA)
print("  ")
print(" ---> Solve A^T A x =  b Using the QR Factorization ----------------")
print(" ---> From A^T A = Q R; we have Q R x =  b ==> R x = Q^T b -------------------")
print("  ")
print(" -----> 2nd: Solve R x = Q^T b  --> x ")
x = solve(R, Q.T @ b)
print("Solution Vector x =", x)
print("  ")
print("Verify that A^T A x =  b  ", ATA @ x ) 
print("  ")
###################################################################
print(" -----  Solution of A^T A x = b Using The SVD Factorization  -----")
#
U, sigma, VT = svd(ATA)
Sigma = diagsvd(sigma, n, n)
#
print(" -----  Since A^t A x = U Sigma V^t x = b ----- ")
print("  ")
print(" -----  1st:  Solve U z = b for z ----- ")
z = U.T @ b
print("  ")
print(" ----- The (n x 1) Vector z -----")
print(z)
print(" ")
print(" ----- 2nd: Solve Sigma y = z  for y ----- ")
y = pinv(Sigma) @ z
print(" ----- The (n x 1) Solution Vector y -----")
print(y)
print(" ")
print(" ----- 3rd: Solve V^t x = y  for x ----- ")
x = VT.T @ y
print(" ----- The (n x 1) Solution Vector x -----")
print(x)
print(" ")
print(" ----- Reconstruct (n x 1) b Vector")
b = ATA @ x
print(b)
print(" ")
########################################################################
print(" -----  Solution of A^T A x = b Using The Pseudo-Inverse -----")
x = pinv(ATA) @ b
print(" ")
print(" ----- The (n x 1) Solution Vector x -----")
print(x)
print(" -------------> END")

