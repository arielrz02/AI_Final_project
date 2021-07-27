import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Preprocess.preprocess_funcs import one_hot_enc
from Preprocess.anomoly_detection_and_PCA import PCA_no_anomolies


def split_data_and_tags(data: pd.DataFrame, train_ratio: int=0.8, tag: str="odor", PCA=False):
    tags = data[tag]
    data = data.drop(tag, axis=1)
    data = one_hot_enc(data)
    if PCA:
        if PCA == True:
            PCA = 20
        data, tags = PCA_no_anomolies(data, tags, n_comp=PCA)
    train_test = train_test_split(data, tags, train_size=train_ratio)
    return train_test
