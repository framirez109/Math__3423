#
# EigenValues Using QR Factorization
#
import numpy as np
from numpy.linalg import qr 
#
A1 = np.array([[0, 5, 1],
               [1, 5, 2],
               [1, 1, 4]], float)
print("A1 = ")
print(A1)

MaxIter = 60; p = [1, 10, 20, MaxIter]
A = A1
for i in range(MaxIter):
    Q, R = qr(A)
    A = R @ Q

    if i+1 in p:
        print(f'Iteration {i+1}:')
        print(R)

print("A = ")
print(A)
print(" ")
print("A1 = ")
print(A1)

