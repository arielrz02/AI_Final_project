import pandas as pd
from typing import Union
import numpy as np

DATA_PATH = "raw_data/"

"""
Taking the data from the file.
"""
def data_to_df(filename="mushrooms_data.txt") -> pd.DataFrame:
    # creating the full path.
    full_file_name = DATA_PATH + filename
    # translating each column in the file to the according feature in the dataframe
    df = pd.read_csv(full_file_name, names=["class", "cap-shape", "cap-surface", "cap-color", "bruises?", "odor",
                                             "gill-attachment", "gill-spacing", "gill-size", "gill-color",
                                             "stalk-shape", "stalk-surface-above-ring",
                                             "stalk-surface-below-ring", "stalk-color-above-ring",
                                             "veil-type", "veil-color", "ring-number", "ring-type",
                                             "spore-print-color", "population", "habitat"],
                     index_col=False)
    return df


"""
One hot encoding the data.
"""
def one_hot_enc(df: Union[pd.DataFrame, pd.Series])->pd.DataFrame:
    one_hot_df = pd.DataFrame()
    # one hot encoding every column separately and connecting.
    if type(df) == pd.DataFrame:
        for col in df:
            temp = pd.get_dummies(df[col])
            one_hot_df = pd.concat([one_hot_df, temp], axis=1)
    # or just encoding for a series.
    else:
        one_hot_df = pd.get_dummies(df)
    return one_hot_df


"""
Translating the letters of the odor to numbers.
"""
def odor_to_tag(df: pd.DataFrame) -> pd.DataFrame:
    odor_dict = {'a': 0, 'l': 1, 'c': 2, 'y': 3, 'f': 4, 'm': 5, 'n': 6, 'p': 7, 's': 8, '-': np.nan}
    df["odor"] = [odor_dict[x] for x in df["odor"]]
    return df
