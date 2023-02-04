import numpy as np
import math

def SOR(n, A, b, Xo, w, TOL, N):
    NumberOfIt = 1
    while (NumberOfIt <= N):
        Xcur = np.zeros(n)
        for i in range(0, n):
            helper = 0
            for j in range(0, i):
                helper += A[i][j] * Xcur[j]
            for j in range(i + 1, n):
                helper += A[i][j] * Xo[j]
            if A[i][i] == 0:
                return "A matrix in not valid"
            Xcur[i] = (1 - w) * Xo[i] + w / A[i][i] * (b[i] - helper)
        # print(Xcur)
        mx = abs(Xcur[0] - Xo[0])
        for i in range(1, n):
            mx = max(mx, Xcur[i] - Xo[i])
        if mx < TOL:
            return Xcur
        Xo = Xcur
        NumberOfIt += 1
    return "number of iterations was exceeded"

def _():
    n = int (input("the number of equations and unknowns "))
    A = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            A[i][j] = float (input ("{i}, {j} entry of matrix ".format(i = i + 1, j = j + 1)))
    b = np.zeros(n)
    for i in range(0, n):
        b[i] = float(input ("{i} entry of b ".format(i = i + 1)))
    Xo = np.zeros(n)
    for i in range(0, n):
        Xo[i] = float (input ("{i} entry of Xo ".format(i = i + 1)))
    w = float (input ("parameter w "))
    TOL = float (input ("tolerance TOL "))
    N = int (input ("Maximum number of iterations "))
    print (SOR(n, A, b, Xo, w, TOL, N))

if __name__ == "__main__":
    # test
    # n = 3
    # A = [[10, -1, 0],
    #      [-1, 10, -2],
    #      [0, -2, 10]]
    # b = [9, 7, 6]
    # Xo = [0, 0, 0]
    # w = 1.2
    # TOL = 0.001
    # N = 10
    # print(SOR(n, A, b, Xo, w, TOL, N))
    # task 3
    # _()
    # task 4
    n = 80
    A = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                A[i][j] = 2 * (i + 1)
            elif abs(i - j) == 2:
                A[i][j] = 0.5 * (i + 1)
            elif abs(i - j) == 4:
                A[i][j] = 1/4 * (i + 1)
    Xo = np.zeros(80)
    b = np.zeros(80)
    for i in range(0, 80):
        Xo[i] = math.pi
    w = 1
    TOL = 0.00001
    print(SOR(n, A, b, Xo, w, TOL, 30))