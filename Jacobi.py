# solve any Ax=b
#1 A - N*N  and invertible
#2 A - M*N and not invertible

# iterative method is X (k+1) = X (k) * B + f
# we have to choose B matrix and f vector to approxiamate solution of Ax=b

# B should be Invertible
# B spectral radius should be >  1 , maximum eigenvalue > 1 and eigenvalues are nonzero

# counting spectral radius, evaluate all eigenvalues
# take B matrix 2-norm is the largest absolute column sum of the matrix

# If A∈R (n,n) and AT=A then ∥A∥2=ρ(A)
# ∥A∥2 ≤ ( ∥AT*A∥∞)^(1/2) - which is simply the maximum absolute row sum of the matrix.


# COMPUTE

# diagonal part of A matrix = D
# non-diagonal part of A matrix = R = L + U

# B = (- D(-1) * R ) check if it is row diagonnaly dominant
# check spectral radius > 1

# iteration
# X (k+1) = X (k) * (- D(-1) * R ) + D (-1) * b

### 450*450 square Matrix with Integer points
# import ImageTesting
# A=ImageTesting.squareImage("test.jpg")
# print(A.shape)
# import numpy as np
# # Convert to 2D matrix by taking the average of the 3D vector
# matrix_2d = np.mean(A, axis=2)
# # Convert to int
# matrix_2d = np.round(matrix_2d * 255).astype(np.uint8)
# print(matrix_2d)
#
# b = np.ones(450)*255
# print(b)

# # D = diagonal matrix of A
# import numpy
# A = matrix_2d
# D = np.diagflat(np.diag(A))
# print(numpy.linalg.det(D))
# D_1 = numpy.linalg.inv(D)
# print(D)
# print(D_1)
# # R = non-diagonal values of A
# #R =


from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot
import numpy as np

def jacobi(A,b,N=25,x=None):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed
    if x is None:
        x = zeros(len(A[0]))

    D = diag(A)
    R = A - diagflat(D)

    # Iterate for N times
    for i in range(N):
        if np.linalg.norm(x - (b - dot(R,x)) / D,np.inf) < 0.001:
            print("XZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZz")
        x = (b - dot(R,x)) / D
        print(x)
    return x

A = array([[10,5,0,0],[5,10,-4,0],[0,-4,8,-1],[0,0,-1,5]])
b = array([6,25,-11,-11])
guess = array([0,0,0,0])

sol = jacobi(A,b,N=10,x=guess)

pprint(A)
pprint(b)
pprint(sol)




