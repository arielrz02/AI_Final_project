import pandas as pd
import numpy as np
from typing import Tuple
from itertools import product
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from Preprocess.Preprocces_whole_data import *
from Preprocess.split_data import split_data_and_tags


def rf_cross_val(train: pd.DataFrame, tags: pd.Series, rfparams: dict, fold=5):
    con_mat_nlst = []
    con_mat_lst = []
    f1_lst = []
    for i in range(fold):
        train_and_val = train_test_split(train, tags, test_size=1/fold)
        conmatn, conmat, f1 = rf_single_hyperparams(*train_and_val, rfparams=rfparams)
        con_mat_nlst.append(conmatn)
        con_mat_lst.append(conmat)
        f1_lst.append(f1)
    con_mat_nlst = np.array(con_mat_nlst)
    mean_nmat = con_mat_nlst.mean(axis=0)
    std_nmat = con_mat_nlst.std(axis=0)

    con_mat_lst = np.array(con_mat_lst)
    mean_mat = con_mat_lst.mean(axis=0)
    std_mat = con_mat_lst.std(axis=0)

    f1_lst = np.array(f1_lst)
    mean_f1 = f1_lst.mean(axis=0)
    std_f1 = f1_lst.std(axis=0)
    return mean_nmat, std_nmat, mean_mat, std_mat, mean_f1, std_f1


def rf_single_hyperparams(train: pd.DataFrame, test: pd.DataFrame, train_tag: pd.Series,
                          test_tag: pd.Series, rfparams: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rf_single_model = RandomForestClassifier(bootstrap=True, n_jobs=-1, **rfparams)
    rf_model = OneVsRestClassifier(rf_single_model).fit(train, train_tag)
    prediction = rf_model.predict(test)
    return evaluate(prediction, test_tag)


def evaluate(prediction, tag) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    con_mat_n = confusion_matrix(prediction, tag, labels=range(9), normalize="true")
    con_mat = confusion_matrix(prediction, tag, labels=range(9), normalize=None)
    f1 = f1_score(prediction, tag, labels=range(9), average="micro")
    return con_mat_n, con_mat, f1

def choose_rf_params(df: pd.DataFrame, tags: pd.Series):
    n_estimators_lst = [int(x) for x in np.linspace(start=100, stop=1300, num=7)]
    max_features_lst = ['auto', 'sqrt']
    max_depth_lst = [int(x) for x in np.linspace(10, 100, num=10)]
    min_split_lst = [2, 5, 10]
    min_leaf_lst = [1, 2, 4]

    maxconmatn = np.zeros((2, 2))
    maxconmat = np.zeros((2, 2))
    maxf1 = np.array(0)

    for n_est, max_feat, max_depth, min_splt, min_leaf in tqdm(product(n_estimators_lst,
        max_features_lst, max_depth_lst, min_split_lst, min_leaf_lst), total=1260):

        paramsgrid = {"n_estimators": n_est, "max_features": max_feat, "max_depth": max_depth,
                      "min_samples_split": min_splt, "min_samples_leaf": min_leaf}

        mean_nmat, std_nmat, mean_mat, std_mat, mean_f1, std_f1 = rf_cross_val(df, tags, paramsgrid)

        if mean_nmat.trace() > maxconmatn.trace():
            maxconmatn = mean_nmat
            maxconmatnstd = std_nmat
            conmatnparams = paramsgrid

        if mean_mat.trace() > maxconmat.trace():
            maxconmat = mean_mat
            maxconmatstd = std_mat
            conmatparams = paramsgrid

        if mean_mat.sum() > maxf1.sum():
            maxf1 = mean_f1
            maxf1std = std_f1
            f1params = paramsgrid

    return maxconmatn, maxconmatnstd, conmatnparams, maxconmat, maxconmatstd, conmatparams, maxf1, maxf1std, f1params



if __name__ == "__main__":
    df = data_to_df("mushrooms_data.txt")
    df = odor_to_tag(df)
    train, test, train_tag, test_tag = split_data_and_tags(df)
    #res = rf_single_hyperparams(train, test, train_tag, test_tag, {'n_estimators': 100, 'max_features': 'auto', 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 4})
    #print("")
    params = choose_rf_params(train, train_tag)
    print(f"max of normalized confusion matrix with params {params[2]}")
    print(f"max of confusion matrix with params {params[5]}")
    print(f"max of f1 with params {params[8]}")
    print(rf_single_hyperparams(train, test, train_tag, test_tag, params[2]))
    print(rf_single_hyperparams(train, test, train_tag, test_tag, params[5]))
    print(rf_single_hyperparams(train, test, train_tag, test_tag, params[8]))
