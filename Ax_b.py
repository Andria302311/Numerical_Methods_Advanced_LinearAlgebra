


def SolveLarge_Ax_b(A,b):
    #GMRES
    import numpy as np
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import gmres
    A = csc_matrix(A, dtype=float)
    b = np.array(b, dtype=float)
    from scipy.sparse.linalg import gmres
    return gmres(A,b)