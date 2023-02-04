import numpy as np

### Construct Block Matrix
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.array([[9, 10], [11, 12]])
D = np.array([[13, 14], [15, 16]])

# Convert A, B, C, and D into a block matrix using the numpy.block function
block_matrix = np.block([[A, B], [C, D]])
print(f'Block matrix:\n{block_matrix}')


### Matrix M*N -> Matrix N*N (m > n)
def toSquareMatrix(A):
    m, n = A.shape[0],A.shape[1]
    if(m>n):
        return A[:n][:n]
    return A[:m][:m]


### Matrix -> Diagonal Matrix
def toDiagonalMatrix(A):
    A=toSquareMatrix(A)
    num_rows, num_cols = A.shape
    D = np.zeros((num_rows, num_cols))
    D[np.arange(num_rows), np.arange(num_cols)] = np.diagonal(A)

    return D

A = np.array([[1, 2], [3, 4],[5,6]])
print(toDiagonalMatrix(A))

### Matrix = L*U decomposition and obtain UpperDiagonalMatrix and LowerDiagonalMatrix

def LU_decomposition(Matrix):
    # obatin L and U matrix
    import scipy.linalg
    P, L, U = scipy.linalg.lu(A)

    return L,U






















# Define a matrix A
A = np.array([[1, 2], [3, 4]])

# Compute the eigenvalues and eigenvectors of A
eigenvalues, eigenvectors = np.linalg.eig(A)

# Define a diagonal matrix D with the eigenvalues of A on the diagonal
D = np.diag(eigenvalues)

# Define a matrix P with the eigenvectors of A as columns
P = eigenvectors

# Compute the inverse of P
P_inv = np.linalg.inv(P)

# Compute the matrix B = P^(-1) * D * P
B = np.dot(P_inv, np.dot(D, P))

print(f'Matrix B:\n{B}')