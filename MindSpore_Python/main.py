import scipy.io
from scipy.spatial.distance import cdist
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
import time

def StandardNyst(pt, landmarks, sigma, dim):
    dis = cdist(landmarks, landmarks)
    K_c = np.exp(-np.square(dis) / (2 * sigma ** 2))
    dis = cdist(pt, landmarks)
    K_all2c = np.exp(-np.square(dis) / (2 * sigma ** 2))
    val, vec = np.linalg.eig(K_c)

    idx = np.argsort(-val)
    vec = vec[:, idx[:dim]]
    emb = np.dot(K_all2c, vec)
    return emb

def runCalErr(embGT, emb):
    signs = np.sign(np.sum(embGT * emb, axis=0))
    emb_aligned = emb * signs
    err = np.linalg.norm(embGT - emb_aligned, 'fro')
    return err

def optimization_loop_2SS(cen0, labelKM, K1, V1, X1, sigma, maxIter, data):

    m1 = X1.shape[0]
    m2 = cen0.shape[0]
    label = labelKM
    cen = cen0.copy()
    iter_count = 1
    thr_cen = 1e-3              # Convergence tolerance tau
    diff_cen = 1e5              # Initialize convergence error
    alpha0 = 1e-3               # Gradient descent searching step

    # Build sparse assignment matrix F_sparse
    _, last = np.unique(label, return_inverse=True)
    row_idx = last
    col_idx = np.arange(m1)
    data_vals = np.ones(m1)
    F_matrix = coo_matrix((data_vals, (row_idx, col_idx)))
    F_sparse = F_matrix.tocsr()

    # Precompute matrices for gradient
    VVF = V1 @ (V1.T @ F_sparse.T)
    P = 2 * F_sparse @ VVF
    Q = 2 * K1 @ VVF

    while diff_cen > thr_cen and iter_count <= maxIter:
        cen_last = cen.copy()
        dFdZij = gradZ_input_Fast(data, cen, sigma, P, Q)
        sfactor = np.linalg.norm(dFdZij, 'fro') / np.linalg.norm(cen, 'fro')
        alpha = alpha0 / sfactor if sfactor > 1 else alpha0
        cen -= alpha * dFdZij
        diff_cen = np.linalg.norm(cen_last - cen, 'fro') / np.linalg.norm(cen_last, 'fro')
        iter_count += 1

    return cen, label, F_sparse, iter_count

def gradZ_input_Fast(data, cen, sigma, P, Q):
    n_features = data.shape[1]
    n_clusters = cen.shape[0]

    # Compute kernel matrix between data and centers
    dis = cdist(data, cen)
    Kz = np.exp(-np.square(dis) / (2 * sigma ** 2))

    # Compute gradient with respect to Kz
    dFdKz = Kz @ P - Q

    # Initialize gradient matrix
    dFdZij = np.zeros((n_features, n_clusters))

    # Compute element-wise product for gradient calculation
    hh = Kz * dFdKz / sigma ** 2

    # Calculate gradient for each cluster center
    for j in range(n_clusters):
        dFdZij[:, j] = (data - cen[j, :]).T @ hh[:, j]

    return dFdZij.T

def run2SSEmb(data, cenOur, FV1, sigma):
    dis = cdist(data, cenOur)
    Kz = np.exp(-np.square(dis) / (2 * sigma ** 2))
    emb = Kz @ FV1
    return emb


def runKMeans_Nystrom(dataTrain, dataCap, m1, sigma, dim, embGT, tolerance=1e-5, n_init=5, max_iter=300):
    n_clusters = m1
    km = KMeans(n_clusters=n_clusters, tol=tolerance, n_init=n_init, max_iter=max_iter)
    km.fit(dataTrain)
    cenKM = km.cluster_centers_
    embKM = StandardNyst(dataCap, cenKM, sigma, dim)
    errKM = runCalErr(embGT, embKM)
    return (cenKM, errKM)

def run2SS_Nystrom(dataTrain, dataCap, m1, m2, sigma, dim, embGT, tolerance=1e-5, n_init=5, max_iter=300):
    # First stage: k-means to get the large landmark set
    km1 = KMeans(n_clusters=m1, tol=tolerance, n_init=n_init, max_iter=max_iter)
    km1.fit(dataTrain)
    cenKM = km1.cluster_centers_

    # Compute kernel matrix and its eigendecomposition
    dis = cdist(cenKM, cenKM)
    Ke = np.exp(-np.square(dis) / (2 * sigma ** 2))
    val, V1 = np.linalg.eigh(Ke)
    idx = np.argsort(-val)
    V1 = V1[:, idx[:dim]]

    # Compute K1
    dis = cdist(dataTrain, cenKM)
    K1 = np.exp(-np.square(dis) / (2 * sigma ** 2))

    # Second stage: k-means to initialize the small landmark set, or z in Alg. 1 of [1]
    # In Alg. 1 of [1], cenKM2SS is represented as z^{(0)}
    km2 = KMeans(n_clusters=m2, tol=tolerance, n_init=n_init, max_iter=max_iter)
    km2.fit(dataTrain)
    cenKM2SS = km2.cluster_centers_

    # Assign each first-stage center to nearest second-stage center
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(cenKM2SS)
    _, labelLast = nn.kneighbors(cenKM, n_neighbors=1)
    labelLast = labelLast.flatten()

    maxIter = 50
    # Weighted clustering refinement
    cen2SS, _, F_sparse, _ = optimization_loop_2SS(
        cenKM2SS, labelLast, K1, V1, cenKM, sigma, maxIter, dataTrain
    )
    
    FV1 = F_sparse @ V1

    # Embedding and error
    emb2SS = run2SSEmb(dataCap, cen2SS, FV1, sigma)
    err2SS = runCalErr(embGT, emb2SS)
    return cen2SS, err2SS


if __name__ == "__main__":
    # load data
    mat = scipy.io.loadmat('CCC_Capital.mat')
    
    # load parameters
    sizeTraining = 500           # Number of samples for training
    m1 = 200                     # size of the large landmark set
    m2 = 30                      # size of the small landmark set
    
    dataCap = mat['dataCap']     # Data matrix (samples x features)
    idxCap = mat['idxCap']       # Ground truth labels

    numData = dataCap.shape[0]   # Number of data samples
    dimData = dataCap.shape[1]   # Feature dimension of data
    dim = len(np.unique(idxCap)) # Number of clusters / embedding dimension

    # Radomly select data for low-rank decomposition
    dataTrain = random.sample(list(dataCap), sizeTraining)
    dataTrain = np.array(dataTrain)

    # Take the eigenvectors of the training data as the ground truth
    dis = cdist(dataTrain, dataTrain)
    # Gaussian scale parameter sigma is set to be the median of all distances
    sigma = np.median(dis)
    # Kernel matrix, or the Gram matrix, of the training data
    Ke = np.exp(-np.square(dis) / (2 * sigma ** 2))
    val, vecGT = np.linalg.eig(Ke)
    idx = np.argsort(-val)
    vecGT = vecGT[:, idx[:dim]]

    # Take the embeddings generated from the training points as the ground truth
    dis = cdist(dataCap, dataTrain)
    K1 = np.exp(-np.square(dis) / (2 * sigma ** 2))
    embGT = np.dot(K1, vecGT)

    # 1. K-means Nystrom
    cenKM, errKM = runKMeans_Nystrom(dataTrain, dataCap, m1, sigma, dim, embGT)

    # 2. 2SS Nystrom
    cen2SS, err2SS = run2SS_Nystrom(dataTrain, dataCap, m1, m2, sigma, dim, embGT)

    print('Error of k-means is ', errKM, ', error of 2SS is ', err2SS)
