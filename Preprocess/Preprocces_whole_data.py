import pandas as pd

DATA_PATH = "../raw_data/"

def data_to_df(filename="mushrooms_data.txt") -> pd.DataFrame:
    full_file_name = DATA_PATH + filename
    df = pd.read_csv(full_file_name, names=["class", "cap-shape", "cap-surface", "cap-color", "bruises?", "odor",
                                             "gill-attachment", "gill-spacing", "gill-size", "gill-color",
                                             "stalk-shape", "stalk-surface-above-ring",
                                             "stalk-surface-below-ring", "stalk-color-above-ring",
                                             "veil-type", "veil-color", "ring-number", "ring-type",
                                             "spore-print-color", "population", "habitat"],
                     index_col=False)
    return df

if __name__ == "__main__":
    data_to_df("mushrooms_data.txt")
