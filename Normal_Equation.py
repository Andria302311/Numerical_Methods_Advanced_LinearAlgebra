# P1
import numpy as np
import numpy.linalg as m
def normale(A, b):
    AT = np.transpose(A)
    # print(m.det(np.matmul(AT,A)))
    if m.det(np.matmul(AT,A)) == 0:
        return "Error"
    return np.matmul(np.matmul(m.inv(np.matmul(AT,A)),AT),b)

# print(normale([[1,3],[1,2]],[0,0]))

def backsubstitution(U, b): # Ux=b
    n = len(U)
    x = np.zeros((n))
    x = x+b
    x[n-1] = b[n-1] / U[n-1][n-1]
    for i in range(n-2,-1,-1):
        for j in range(i+1,n):
            x[i] = (x[i] - U[i][j]*x[j])
        x[i] = x[i] / U[i][i]
    return x

def forwardsubstitution(L, b): # Ux=b
    n = len(L)
    x = np.zeros((n))
    x = x+b
    x[0] = b[0] / L[0][0]
    for i in range(1,n):
        for j in range(i):
            x[i] = x[i] - L[i][j] * x[j]
        x[i] = x[i] / L[i][i]
    return x

def laststep(A, b):
    return normale(A,b)

def _():
    for i in range(0, 100):
        A = np.random.randint(100, size = (11, 11))
        x = np.random.randint(100, size = 11)
        b = np.matmul(A, x)
        b[0] += 1
        b[1] -= 1
        x1 = laststep(A, b)
        x2 = x1
        x3 = x1
        print(i, np.linalg.norm(x - x1, np.inf), np.linalg.norm(x - x2, np.inf), np.linalg.norm(x - x3, np.inf))

print(_())