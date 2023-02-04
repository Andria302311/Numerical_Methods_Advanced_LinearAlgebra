# P2
import numpy as np
import numpy.linalg as m

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



def scal(alfa,v):
    v1 = np.zeros(len(v))
    for i in range(len(v)):
        v1[i] = alfa * v[i]
    return v1

def MGS(mat1):
    mat = np.zeros((len(mat1),len(mat1[0])))
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            mat[i][j] = mat1[i][j]
    Q = np.zeros((len(mat),len(mat[0])))
    R = np.zeros((len(mat[0]), len(mat[0])))
    for j in range(0,len(mat[0])):
        sumsq = 0
        for i in range(len(mat)):
            sumsq += mat[i][j] ** 2
            Q[i][j] = mat[i][j]
        sumsq = sumsq ** (1 / 2)
        R[j][j] = sumsq
        if sumsq < 1e-14:
            return "Error"
        for i in range(len(mat)):
            Q[i][j] /= sumsq
        for k in range(j + 1, len(mat[0])):
            R[j][k] = np.dot(np.transpose(mat)[k], np.transpose(Q)[j])
            v = np.transpose(mat)[k] - R[j][k] * np.transpose(Q)[j]
            for i in range(0, len(mat)):
                mat[i][k] = v[i]
    return Q,R


def mergeofmats(E,F):
    G = np.zeros((len(E)+len(F), len(E[0])))
    for i in range(len(E)+len(F)):
        for j in range(len(E[0])):
            if i <len(E):
                G[i][j] = E[i][j]
            else:
                G[i][j] = F[i-len(E)][j]
    return G
def dividemats(E, m):
    Q1 = np.zeros((m, len(E[0])))
    Q2 = np.zeros((len(E)-m, len(E[0])))
    for i in range(len(E)):
        for j in range(len(E[0])):
            if i<m:
                Q1[i][j] = E[i][j]
            else:
                Q2[i-m][j] = E[i][j]
    return Q1,Q2
def mod(t):
    c = 2
    for i in range(len(t)):
        c = min(c,t[i])
    for j in range(len(t)):
        t[j] += 2 * abs(c)
    return t
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

# print(mergeofmats([[1,2],[3,4]],[[5,6],[7,8]]))
# print(dividemats([[1,2],[3,4],[5,6]],2))
def kkt(A,b, C,d):
    D = mergeofmats(A,C)
    QD, RD = MGS(D)
    q1, q2 = dividemats(QD, len(A))
    q2TQ, q2TR = G(np.transpose(q2))
    u = forwardsubstitution(np.transpose(q2TR), d)
    c = np.matmul(np.matmul(np.transpose(q2TQ), np.transpose(q1)), b) - u
    w = backsubstitution(q2TR, c)
    y = np.matmul(np.transpose(q1), b) - np.matmul(np.transpose(q2), w)
    return backsubstitution(RD, y)
def _():
    C = np.random.randint(1000, size = (8, 10))
    x = np.random.randint(1000, size = 10)
    A = np.random.randint(1000, size = (20, 10))
    b = np.matmul(A, x)
    b[0] += 1
    b[1] -= 1
    d = np.matmul(C, x)
    x1 = kkt(A, b, C, d)
    print(mod(x1))
print(_())
