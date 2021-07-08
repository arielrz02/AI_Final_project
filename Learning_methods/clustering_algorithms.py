import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

from Preprocess.Preprocces_whole_data import data_to_df, one_hot_enc
from create_distances import create_dist_mat
from Plot.dim_reduction_plotting import MDS_and_plot, PCA_and_plot


def spectral_cluster(X: pd.DataFrame, n_clusters=9, **kwargs) -> list:
    clustering = SpectralClustering(n_clusters=n_clusters, **kwargs).fit(X)
    return clustering.labels_


def using_Kmeans(X: pd.DataFrame, n_clusters=9, **kwargs) -> list:
    clustering = KMeans(n_clusters=n_clusters, **kwargs).fit(X)
    return clustering.labels_


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
    tagSC = list(spectral_cluster(df, n_clusters=15))
    tagKM = list(using_Kmeans(df, n_clusters=15))
    PCA_and_plot(df, labels=tag, title="odor_based")
    PCA_and_plot(df, labels=tagSC, title="spectral_cluster")
    PCA_and_plot(df, labels=tagKM, title="KMeans")
