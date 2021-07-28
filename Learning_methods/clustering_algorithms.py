import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


"""
Clustering using the KMenas algorithm.
    X: The data to cluster.
    n_clusters: the number of clusters.
    can except other arguments.
    :returns clustering labels.
"""
def using_spectral_cluster(X: pd.DataFrame, n_clusters=15, **kwargs) -> list:
    clustering = SpectralClustering(n_clusters=n_clusters, **kwargs).fit(X)
    return clustering.labels_


"""
Clustering using the KMenas algorithm.
    X: The data to cluster.
    n_clusters: the number of clusters.
    can except other arguments.
    :returns clustering labels.
"""
def using_Kmeans(X: pd.DataFrame, n_clusters=15, **kwargs) -> list:
    clustering = KMeans(n_clusters=n_clusters, **kwargs).fit(X)
    return clustering.labels_


"""
Clustering using the DBSCAN algorithm.
    X: The data to cluster.
    epsilon: the radius for core points.
    can except other arguments.
    :returns clustering labels.
"""
def using_dbscan(X: pd.DataFrame, eps=1.5) -> list:
    clustering = DBSCAN(eps=eps).fit(X)
    return clustering.labels_


"""
Clustering using the GMM algorithm.
    X: The data to cluster.
    n_clusters: the number of clusters.
    tol: The convergence threshold.
    can except other arguments.
    :returns clustering labels.
"""
def using_GMM(X: pd.DataFrame, n_clusters=12, tol=0.001, **kwargs) -> list:
    clustering = GaussianMixture(n_components=n_clusters, tol=tol, **kwargs).fit(X)
    return clustering.predict(X)


"""
Computes the silhouette score.
    X: the clustered data.
    labels: the clustering labels
"""
def sil_score(X: pd.DataFrame, labels:list) -> float:
    return silhouette_score(X=X, labels=labels)




