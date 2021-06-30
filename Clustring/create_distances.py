import pandas as pd
from tqdm import tqdm
from AI_Final_project.Preprocess.Preprocces_whole_data import data_to_df

def create_dist_matrix(df: pd.DataFrame, exponent=1) -> pd.DataFrame:
    distances = pd.DataFrame(0, index=df.index, columns=df.index, dtype=float)
    for row in tqdm(distances.index, desc="creating distance for samples"):
        for col in range(row):
            dist = get_dist(df.loc[row, :], df.loc[col, :], exponent)
            distances.loc[row, col] = distances.loc[col, row] = dist
    return distances


def get_dist(obj1: pd.Series, obj2: pd.Series, exp=1) -> float:
    dist = 0
    for e1, e2 in zip(obj1.values, obj2.values):
        if e1 == e2:
            dist += 1
    return dist ** (1/exp)

if __name__ == "__main__":
    df = data_to_df("mushrooms_data.txt")
    create_dist_matrix(df)