import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture

from Preprocess.preprocess_funcs import data_to_df, one_hot_enc
from Plot.dim_reduction_plotting import PCA_and_plot


def using_spectral_cluster(X: pd.DataFrame, n_clusters=15, **kwargs) -> list:
    clustering = SpectralClustering(n_clusters=n_clusters, **kwargs).fit(X)
    return clustering.labels_


def using_Kmeans(X: pd.DataFrame, n_clusters=15, **kwargs) -> list:
    clustering = KMeans(n_clusters=n_clusters, **kwargs).fit(X)
    return clustering.labels_


def using_dbscan(X: pd.DataFrame, eps=1.5) -> list:
    clustering = DBSCAN(eps=eps).fit(X)
    return clustering.labels_

def using_GMM(X: pd.DataFrame, n_clusters=12, tol=0.001, **kwargs) -> list:
    clustering = GaussianMixture(n_components=n_clusters, tol=tol, **kwargs).fit(X)
    return clustering.predict(X)


if __name__ == "__main__":
    odor_dict = {'a': 0, 'l': 1, 'c': 2, 'y': 3, 'f': 4, 'm': 5, 'n': 6, 'p': 7, 's': 8}
    df = data_to_df("mushrooms_data.txt")
    df = df.sample(frac=1.0)
    tag = df["odor"]
    df = df.drop(["odor"], axis=1)
    tag = [odor_dict[x] for x in tag]
    #tag = one_hot_enc(tag)
    df = one_hot_enc(df)
    #mat = create_dist_mat(df)
    #tagSC = using_spectral_cluster(df, n_clusters=15)
    #tagKM = using_Kmeans(df, n_clusters=15)
    tagDB = using_dbscan(df, eps=1.5)
    #tagGMM = using_GMM(df, n_clusters=12, tol=0.001)
    #PCA_and_plot(df, labels=tag, title="odor_based")
    #PCA_and_plot(df, labels=tagSC, title="spectral_cluster")
    #PCA_and_plot(df, labels=tagKM, title="KMeans")
    PCA_and_plot(df, labels=tagDB, title="DBSCAN", folder="plots")
    #PCA_and_plot(df, labels=tagGMM, title="GMM", folder="plots")


