import numpy as np

def modified_gram_schmidt(vectors):
    orthonormal_vectors = []
    for i in range(len(vectors)):
        v = vectors[i]
        for u in orthonormal_vectors:
            v = v - np.dot(v, u) * u
        norm = np.linalg.norm(v)
        if np.isclose(norm, 0):  # check if the norm of the vector is zero
            return orthonormal_vectors
        orthonormal_vectors.append(v / norm)
    return orthonormal_vectors


vectors = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
orthonormal_vectors = modified_gram_schmidt(vectors)
print(orthonormal_vectors)

def IsOrthonormal(vectors):
    for i in range(len(orthonormal_vectors)):
        if(np.isclose(np.linalg.norm(orthonormal_vectors[i]),1,atol=1e-7)):
            continue
        return False
    return True

print(IsOrthonormal(vectors))
