from argparse import ArgumentParser
import pandas as pd

from Preprocess.preprocess_full import prep_missing_data, prep_whole_data
from Learning_methods.random_forest import choose_rf_params, rf_single_hyperparams
from Learning_methods.multiclass_NN import running_nni, neural_net
from Learning_methods.clustering_algorithms import *
from Plot.dim_reduction_plotting import PCA_and_plot

def main(args):
    data = args.data
    if data == "whole":
        train, test, train_tags, test_tags = prep_whole_data()
    elif data == "missing":
        train, test, train_tags, test_tags = prep_missing_data()
    else:
        print(f"Data type {data} isn't a valid type.")
        return
    findParams = args.findParams
    model = args.model
    if findParams == "true":
        if model == "random_forest":
            res = choose_rf_params(train, train_tags)
            print(res)
        elif model == "neural_network":
            running_nni(train, test, train_tags, test_tags)
        else:
            print(f"The model {model} doesn't have parameter a maximization function")
            return
    elif findParams == "false":
        if model == "KMeans" or model == "DBSCAN" or model == "GMM" or\
                model == "spectral_clustering" or model == "odor_based":
            data = train.append(test)
            if model == "KMeans":
                tags = using_Kmeans(data)
            elif model == "spectral clustering":
                tags = using_spectral_cluster(data)
            elif model == "GMM":
                tags = using_GMM(data)
            elif model == "DBSCAN":
                tags = using_dbscan(data)
            else:
                tags = train_tags.append(test_tags)
            PCA_and_plot(data, labels=tags, title=model)
        elif model == "random_forest":
            res = rf_single_hyperparams(train, test, train_tags, test_tags,
                                        {'n_estimators': 500, 'max_features': 'sqrt',
                                         'max_depth': 20, 'min_samples_split': 2,
                                         'min_samples_leaf': 4})
            print(res)
        elif model == "neural_network":
            res = neural_net(train, test, train_tags, test_tags, charsize=train.shape[1])
            print(res) #TODO: make priting in the runfile nicer
        else:
            print(f"The model {data} doesn't exist")
            return
    else:
        print(f"find_params can only be true or false")
        return



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='neural_network')
    parser.add_argument('--data', '-d', type=str, default='whole')
    parser.add_argument('--findParams', '-f', type=str, default='false')
    args = parser.parse_args()
    main(args)
