"""Artificial intelligence project by Ariel Rosen-Zvi, ID #325172203"""

from argparse import ArgumentParser

from Preprocess.preprocess_full import prep_missing_data, prep_whole_data
from Learning_methods.random_forest import choose_rf_params, rf_single_hyperparams, rf_cross_val
from Learning_methods.multiclass_NN import running_nni, neural_net
from Learning_methods.clustering_algorithms import *
from Plot.dim_reduction_plotting import PCA_and_plot


"""
The main function that runs the project.
"""
def main(args):
    # choosing whether to use less features using PCA or to not.
    bf = args.better_features
    bf_dict = {"false": False, "true": True}
    if bf != "true" and bf != "false":
        print("better_featuers can only be true or false")
        return
    # choosing data type, either whole or missing.
    dataType = args.data
    if dataType == "whole":
        train, test, train_tags, test_tags = prep_whole_data(Pca_anomolies=bf_dict[bf])
    elif dataType == "missing":
        train, test, train_tags, test_tags = prep_missing_data(Pca_anomolies=bf_dict[bf])
    else:
        print(f"Data type {dataType} isn't a valid type.")
        return
    # choosing if to use the model with my params or
    # to try to find the best parameter using the code.
    findParams = args.findParams
    # chossing learning model.
    model = args.model
    if findParams == "true":
        if model == "random_forest":
            res = choose_rf_params(train, train_tags)
            print(f"the grid search gave the following results:\n"
                  f"the highest f1 micro score we got is {res[0]}, with a standard deviation of {res[1]}.\n"
                  f"it was reached with the following parameters: {res[2]}\n"
                  f"the grid search gave the following results:\n"
                  f"the highest f1 micro score we got is {res[3]}, with a standard deviation of {res[4]}.\n"
                  f"it was reached with the following parameters: {res[5]}\n")
        elif model == "neural_network":
            running_nni(train, test, train_tags, test_tags)
            print("Please make sure you are running this through the NNI interface")
        else:
            print(f"The model {model} doesn't have parameter a maximization function")
            return
    elif findParams == "false":
        # using a clustering algorithm.
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
            # getting the silhouette score and plotting the clusters.
            score = sil_score(data, tags)
            PCA_and_plot(data, labels=tags, title=model+' visualization')
            print(f"The silhouette score is {score}, a visualization was saved in the plots folder.")
        # using random forest.
        elif model == "random_forest":
            res = rf_single_hyperparams(train, test, train_tags, test_tags,
                                        {'n_estimators': 500, 'max_features': 'sqrt',
                                         'max_depth': 20, 'min_samples_split': 2,
                                         'min_samples_leaf': 4})
            print(f"the f1 micro score is {res[0]} and the f1 macro score is {res[1]}")
        # using the neural network.
        elif model == "neural_network":
            res = neural_net(train, test, train_tags, test_tags, charsize=train.shape[1])
            print(f"the f1 micro score is {res[0]} for the test and {res[1]} for the training data\n"
                  f"the f1 macro score is {res[2]} for the test and {res[3]} for the training data\n")
        else:
            print(f"The model {model} doesn't exist")
            return
    else:
        print("find_params can only be true or false")
        return



if __name__ == "__main__":
    # taking the arguments from the command line.
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='neural_network')
    parser.add_argument('--data', '-d', type=str, default='whole')
    parser.add_argument('--findParams', '-f', type=str, default='false')
    parser.add_argument('--better_features', '-b', type=str, default='false')
    args = parser.parse_args()
    # running the project with the arguments we got.
    main(args)
