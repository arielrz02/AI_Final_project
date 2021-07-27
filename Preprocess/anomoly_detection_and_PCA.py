import pandas as pd
import numpy as np
from itertools import product

from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

from Learning_methods.random_forest import rf_cross_val


def PCA_no_anomolies(data: pd.DataFrame, tags: pd.Series, n_comp=20):
    lof = LocalOutlierFactor(n_neighbors=15, n_jobs=-1)
    anomolies = lof.fit_predict(data)
    data = data[anomolies == 1]
    tags = tags[anomolies == 1]
    pca = PCA(n_components=n_comp)
    short_data = pd.DataFrame(pca.fit(data.T).components_)
    return short_data.T, tags

if __name__=="__main__":
    a=1
    # train, test, train_tags, test_tags = prep_whole_data()
    # data = train.append(test)
    # tags = train_tags.append(test_tags)
    # data, tags = PCA_no_anomolies(data, tags)
    # rf_cross_val()
