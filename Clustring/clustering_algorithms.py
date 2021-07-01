import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from Preprocess.Preprocces_whole_data import data_to_df
from create_distances import create_dist_mat
from Plot.dim_reduction_plotting import MDS_and_plot

def spectral_cluster(X: pd.DataFrame, n_clusters=8, n_init=10, gamma=1) -> list:
    clustering = SpectralClustering(affinity="precomputed", n_clusters=n_clusters,
                                    n_init=n_init, gamma=gamma).fit(X)
    return clustering.labels_


if __name__ == "__main__":
    df = data_to_df("mushroom_small_data.txt")
    tag = df["odor"]
    df = df.drop(["odor"], axis=1)
    mat = create_dist_mat(df)
    MDS_and_plot(mat, labels=tag, title="odor_based")
