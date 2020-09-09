#Import
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

#Functions
def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    index = np.random.randint(0, m, k)
    
    for i in range(k):
        centroids[i,:] = X[index[i],:]
    
    return centroids

def find_closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    index = np.zeros(m)
    
    
    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i,:] - centroids[j,:]) ** 2)
            if dist < min_dist:
                min_dist = dist
                index[i] = j
    
    return index

def compute_centroids(X, index, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    
    for i in range(k):
        indices = np.where(index == i)
        centroids[i,:] = (np.sum(X[indices,:], axis=1) / len(indices[0])).ravel()
    
    return centroids

def k_means(X, k, max_iter):
    m, n = X.shape
    index = np.zeros(m)
    centroids =  init_centroids(X, k)

    for i in range(max_iter):
        index = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, index, k)
    
    return index, centroids

def pca(X):
    X = (X - X.mean()) / X.std()
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]
    U, S, V = np.linalg.svd(cov)
    return U, S, V

def project_data(X, U, k):
    U_reduced = U[:, :k]
    return np.dot(X, U_reduced)


def recovered_data(Z, U, k):
    U_reduced = U[:, :k]
    return np.dot(Z, U_reduced.T)
    
    
    
#Data    
data = loadmat('faces.mat')
X = data['X']
print(X.shape)

face = np.reshape(X[2020, :], (32,32))
plt.imshow(face.T)
plt.show()

U, S, V = pca(X)
Z = project_data(X, U, 100)

X_recovered = recovered_data(Z, U, 100)
recovered_face = np.reshape(X_recovered[2020, :], (32,32))
plt.imshow(recovered_face.T)