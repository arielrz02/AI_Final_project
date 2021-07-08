import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
from Preprocess.Preprocces_whole_data import data_to_df
from Plot.dim_reduction_plotting import MDS_and_plot

def create_dist_mat(df: pd.DataFrame) -> pd.DataFrame:
    length = len(df)
    v = df.values
    distvec = np.array([abs((v[a] - v[b])).sum()
                  for a,b in tqdm(combinations(np.arange(len(df)), 2), total=int((len(df)**2-len(df))/2), desc="creating distances")])
    npmat = np.zeros((length, length))
    npmat[np.triu_indices(length, 1)] = distvec
    npmat = npmat + npmat.transpose()
    pdmat = pd.DataFrame(npmat, index=df.index, columns=df.index)
    return pdmat


