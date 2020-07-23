from sklearn.neighbors import KernelDensity
import numpy as np
from sklearn.neighbors import KDTree


def get_proba(X,n_neighbors=15):
  tree = KDTree(X)         
  D, idx = tree.query(X, k=n_neighbors)
  P = np.zeros([len(X), len(X)])
  for i in range(len(X)):
    kde = KernelDensity().fit(X[idx[i]])
    P[i,:] = np.exp(kde.score_samples(X))
  
  return np.sum(P,axis=1)/np.max(np.sum(P,axis=1))
