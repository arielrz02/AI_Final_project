import pandas as pd
import numbers as np
from sklearn.model_selection import train_test_split

def split_data_and_tags(data: pd.DataFrame, train_ratio: int=0.8, tag: str="odor"):
    tags = data[tag]
    data = data.drop(tag, axis=1)
    train_test = train_test_split(data, tags, train_size=train_ratio)
    return train_test

