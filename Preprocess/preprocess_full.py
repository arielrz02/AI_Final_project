import pandas as pd
import numpy as np

from Preprocess.preprocess_funcs import data_to_df, odor_to_tag
from Preprocess.split_data import split_data_and_tags

def prep_whole_data(file:str="mushrooms_data.txt", Pca_anomolies=False):
    df = data_to_df(file)
    df = odor_to_tag(df)
    train_test_tags = split_data_and_tags(df, PCA=Pca_anomolies)
    return train_test_tags

def prep_missing_data(file:str="mushrooms_data_missing.txt", Pca_anomolies=False):
    df = data_to_df(file)
    df = odor_to_tag(df)
    df.dropna(inplace=True)
    df.replace("-", np.nan, inplace=True)
    train_test_tags = split_data_and_tags(df, PCA=Pca_anomolies)
    return train_test_tags