import scipy.io
from scipy.spatial.distance import cdist
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix


def StandardNyst(pt, landmarks, sigma, dim):
    dis = cdist(landmarks, landmarks)
    K_c = np.exp(-dis*dis/2/sigma**2)
    dis = cdist(pt, landmarks)
    K_all2c = np.exp(-dis*dis/2/sigma**2)
    val, vec = np.linalg.eig(K_c)

    idx = np.argsort(-val)
    vec = vec[:, idx[:dim]]
    emb = np.dot(K_all2c, vec)
    return emb


def runCalErr(embGT, emb):
    for i in range(embGT.shape[1]):
        if np.dot(embGT[:,i].T,emb[:,i])<0:
            emb[:,i] = -emb[:,i]
    err = np.linalg.norm(embGT-emb, 'fro')
    return err

def kernpca_weightedClustering(cen0,labelKM,K1, V1, X1, sigma, maxIter, data):
    m1 = X1.shape[0]
    m2 = cen0.shape[0]
    B = V1
    label = labelKM
    #New Code
    cen = cen0
    iter = 1
    thr_cen = 1e-5
    diff_cen = 1e5
    alpha0 = 1e-3
    junk, last = np.unique(label, return_index=False, return_inverse=True)
    
    _row = last
    _col = np.arange(0, m1)
    _data = np.ones((m1,1)).flatten()
    T_matrix = coo_matrix((_data, (_row, _col)))
    T_sparse = T_matrix.tocsr()

    BBT = B @ (B.T @ T_sparse.T)
    TBBT = 2*T_sparse@BBT
    KBBT = 2*K1@BBT

    while diff_cen>thr_cen and iter<=maxIter:
        last = cen
        dFdUij = gradU_input_Fast(data,cen,sigma,TBBT,KBBT)
        sfactor = np.linalg.norm(dFdUij,'fro')/np.linalg.norm(cen,'fro')

        if sfactor>1:
            alpha = alpha0/sfactor
        else:
            alpha = alpha0

        cen = cen - alpha*dFdUij
        diff_cen = np.linalg.norm(last-cen,'fro') / np.linalg.norm(last,'fro')
        iter = iter+1

    return (cen, label, T_sparse, iter)

def gradU_input_Fast(data,cen,sigma,TBBT,KBBT):
    dim = data.shape[1]
    k = cen.shape[0]

    dis = cdist(data, cen)
    Kz = np.exp(-dis*dis/2/sigma**2)

    dFdKz = Kz@TBBT-KBBT

    dFdUij = np.zeros((dim,k))

    hh = Kz*dFdKz/sigma**2

    for j in range(k):
        dFdUij[:,j] = (data-cen[j,:]).T @ hh[:,j]

    return dFdUij.T

def run2SSEmb(data, cenOur, TV1, sigma):
    dis = cdist(data, cenOur)
    Kz = np.exp(-dis*dis/2/sigma**2)
    emb = Kz @ TV1
    return emb

# load data
mat = scipy.io.loadmat('CCC_Capital.mat')

dataCap = mat['dataCap']
idxCap = mat['idxCap']

numData  = dataCap.shape[0]
dimData = dataCap.shape[1]

# No. of clusters, or the embedding dimension
dim = len(np.unique(idxCap))

# Radomly select data for low-rank decomposition
sizeTraining = 500

idx = list(range(0,numData))

random.shuffle(idx)

dataTrain = dataCap[idx[1:sizeTraining+1]]




dis = cdist(dataTrain, dataTrain)

# Note that np.median is slightly different from the matlab function median
sigma = np.median(dis)


Ke = np.exp(-dis*dis/2/sigma**2)
val, vecGT = np.linalg.eig(Ke)

idx = np.argsort(-val)
vecGT = vecGT[:, idx[:dim]]

dis = cdist(dataCap, dataTrain)
K1 = np.exp(-dis*dis/2/sigma**2)
embGT = np.dot(K1,vecGT)


m1 = 200
m2 = 30


n_clusters = m1
tolerance = 1e-5
max_iter = 300
n_init = 5
km = KMeans(n_clusters=n_clusters, tol=tolerance, n_init=n_init, max_iter=max_iter)

km.fit(dataTrain)
# estimator.labels_ #获取聚类标签
cenKM = km.cluster_centers_ #获取聚类中心
embKM = StandardNyst(dataCap, cenKM, sigma, dim)


errKM = runCalErr(embGT, embKM)

# 2. Our 2SS
# Taking k-means centers as the initial landmarks
X1 = cenKM
dis = cdist(X1,X1)
# learn V1
Ke = np.exp(-dis*dis/2/sigma**2)
val, V1 = np.linalg.eig(Ke)

V1 = V1[:,:dim]

# K1 and K1*V1
dis = cdist(dataTrain,X1)
K1 = np.exp(-dis*dis/2/sigma**2)

n_clusters = m2
tolerance = 1e-5
max_iter = 300
n_init = 5
km = KMeans(n_clusters=n_clusters, tol=tolerance, n_init=n_init, max_iter=max_iter)

km.fit(dataTrain)
cenKM2SS = km.cluster_centers_ #获取聚类中心
labelLast = km.labels_ #获取聚类标签

nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
nn.fit(cenKM2SS)
dist, labelLast = nn.kneighbors(X1, n_neighbors=1)


# main function
cen2SS, junk, T_sparse, junk2 = kernpca_weightedClustering(cenKM2SS,labelLast,K1, V1, X1, sigma,5,dataTrain)
TV1 = T_sparse @ V1

#Embedding
emb2SS = run2SSEmb(dataCap, cen2SS, TV1, sigma)
err2SS = runCalErr(embGT, emb2SS)

print('Error of k-means is ', errKM, ', error of 2SS is ', err2SS)