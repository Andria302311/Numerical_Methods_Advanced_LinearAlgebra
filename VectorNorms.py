import numpy as np

### Hölder norm of x _ P norm
def VectorNormP(vector,p):
    return  np.linalg.norm(vector, ord=p)

def MatrixNormP(vector,Matrix,p):
    return np.linalg.norm(np.dot(Matrix, vector), ord=p)



