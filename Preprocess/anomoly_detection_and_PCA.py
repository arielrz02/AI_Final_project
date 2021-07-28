import pandas as pd

from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA


"""
Removing anomalies and the reducing dimensions using PCA
"""
def PCA_no_anomolies(data: pd.DataFrame, tags: pd.Series, n_comp=20):
    # removing anomalies.
    lof = LocalOutlierFactor(n_neighbors=15, n_jobs=-1)
    anomolies = lof.fit_predict(data)
    data = data[anomolies == 1]
    tags = tags[anomolies == 1]
    # reducing dimensions.
    pca = PCA(n_components=n_comp)
    short_data = pd.DataFrame(pca.fit(data.T).components_)
    return short_data.T, tags

