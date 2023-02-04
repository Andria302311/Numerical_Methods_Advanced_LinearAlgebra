# P1
import numpy as np

def vvT(v):
    mt = np.zeros((len(v), len(v)))
    for i in range(len(v)):
        for j in range(len(v)):
            mt[i][j] = float(v[i]) * float(v[j])
    return mt

def H(mtx):
    mtx = np.matrix(mtx)
    a = np.array(mtx.transpose()[0])[0]
    w = np.linalg.norm(a) * np.eye(len(a))[:, 0]
    v = w - a
    part1 = 2 / np.dot(v, v)
    H = np.identity(len(mtx)) - (part1 * vvT(v))
    return H

def forwardsubstitution(L, b): # Ux=b
    n = len(L[0])
    x = np.zeros((n))
    x[0] = b[0] / L[0][0]
    for i in range(1,len(L)):
        x[i] = b[i]
        for j in range(i):
            x[i] = x[i] - L[i][j] * x[j]
        x[i] = x[i] / L[i][i]
    return x

def backsubstitution(U, b): # Ux=b
    n = len(U[0])
    x = np.zeros((len(U[0])))
    x[n-1] = b[n-1] / U[n-1][n-1]
    for i in range(n-2,-1,-1):
        x[i] = b[i]
        for j in range(i+1,n):
            x[i] = (x[i] - U[i][j]*x[j])
        x[i] = x[i] / U[i][i]
    return x

def G(mtx):
    Q = np.identity(len(mtx))
    for i in range(min(len(mtx)-1, len(mtx[0]))):
        mat2 = np.zeros((len(mtx)-i,len(mtx[0])-i))
        for j in range(i, len(mtx)):
            for k in range(i,len(mtx[0])):
                mat2[j-i][k-i] = mtx[j][k]
        H1 = H(mat2)
        H2 = np.identity(len(mtx))
        for k in range(i,len(mtx)):
            for t in range(i, len(mtx)):
                H2[k][t] = H1[k-i][t-i]
        # print(H2)
        mtx = np.matmul(H2,mtx)
        Q = np.matmul(Q,H2)
    return Q, mtx

def laststep(A, b):
    Q, R = G(A)
    b = np.matmul(np.transpose(Q), b)
    return backsubstitution(R, b)

A = np.random.rand(1000, 1000)
b = np.random.rand(1000)
x = laststep(A, b)

print(x)
print(np.allclose(np.dot(A, x), b))
