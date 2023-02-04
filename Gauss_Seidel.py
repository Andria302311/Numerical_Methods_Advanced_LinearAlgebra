# Gauss-Seidel
import numpy as np
import numpy.linalg as m
from numpy.linalg import inv

def DplusL(A):
    B = np.zeros((len(A),len(A[0])))
    for i in range(len(A)):
        for j in range(len(A[0])):
            if i <= j:
                B[i][j] = A[i][j]
    return B

# def g(A,b, TOL): # Ax=b
#     x0 = np.zeros(len(A))
#     t = 1
#     P = DplusL(A) # P=D+L
#
#
#     minustA = np.zeros((len(A), len(A)))
#     Pinverse = np.linalg.inv(P)
#     for i in range(len(A)):
#         for j in range(len(A)):
#             minustA[i][j] = -t * A[i][j]
#     iminustA = np.identity(len(A)) + minustA
#     PinvtimesiminustA = np.matmul(np.linalg.inv(P), iminustA)
#     tb = np.multiply(b, t)
#     pinvtb = np.matmul(Pinverse, tb)
#     xNext = x0
#     c = []
#     d = 0
#     while True:
#         c = np.matmul(PinvtimesiminustA, xNext) + pinvtb
#         # print(c) #  If we want to get all x(n)
#         if np.linalg.norm(xNext - c, np.inf) < TOL:
#             break
#         xNext = c
#         c = []
#         d += 1
#     return c
#     # print(xNext)
#     # print(c)

from scipy.sparse.linalg import gmres

def g(A, b, TOL):
    x, info = gmres(A, b, tol=TOL)
    if info != 0:
        raise ValueError("GMRES failed to converge. Try increasing the maximum number of iterations or decreasing the tolerance.")
    return x

def dividemat(A, B, TOL):
    ans = []
    div = np.transpose(B)
    for j in range(len(div)):
        ans.append(g(A, np.transpose(div[j]), TOL))
    return np.transpose(ans)



# def dividemat(A, B, TOL): # Ax=B
#     # div = []
#     ans = []
#
#     div = np.transpose(B)
#
#     for j in range(len(div)):
#         ans.append(g(A, np.transpose(div[j]), TOL))
#     # print(div)
#     return np.transpose(ans)

# g([[0.1,0.1],[0.2,0.3]],[0.2,0.5],0.00001)

def _():
    for i in range(0, 1):
        # A = [[0.3,0.05,0,0,0],[0,0.2,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
        A = np.random.randint(3, size=(5, 5))
        # A = np.random.randint(20, size = (5, 5))
        X = np.random.randint(100, size = (5, 2))
        B = np.matmul(A, X)
        print(B)
        Xsol = dividemat(A, B, 1e-9)               #  TEST
        print(Xsol)
        print(X)
        print(i, np.linalg.norm(X - Xsol, np.inf))
        # print(len(Xsol))
        # print(len(Xsol[0]))

print(_())
# I - t P^-1 A   AX = B   P = I - preconditioner

# A = np.random.rand(500, 500)
# x = np.random.rand(500)
# b = np.matmul(A, x)
# x_sol = g(A, b, TOL=1e-9)
# rel_error = np.linalg.norm(x_sol - x) / np.linalg.norm(x)
# print(x)
# print(x_sol)