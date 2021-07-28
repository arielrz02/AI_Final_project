import pandas as pd
from sklearn.model_selection import train_test_split
from Preprocess.preprocess_funcs import one_hot_enc
from Preprocess.anomoly_detection_and_PCA import PCA_no_anomolies

"""
Splitting the data into the train and test data, and the train and test tags.
"""
def split_data_and_tags(data: pd.DataFrame, train_ratio: int = 0.8, tag: str = "odor", PCA = False):
    # splitiing the tags.
    tags = data[tag]
    data = data.drop(tag, axis=1)
    # one hot encoding.
    data = one_hot_enc(data)
    # making the features for the third model.
    if PCA:
        if PCA == True:
            PCA = 20
        data, tags = PCA_no_anomolies(data, tags, n_comp=PCA)
    # splitting to train and test.
    train_test = train_test_split(data, tags, train_size=train_ratio)
    return train_test
