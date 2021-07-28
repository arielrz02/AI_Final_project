import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

"""
plots the clustered data 3D space
"""
def plot_data_3d(X_3d: pd.DataFrame, color: pd.Series, size=10):
    # creating a figure.
    fig = plt.figure()
    ax = Axes3D(fig)
    groups = X_3d.groupby(color)
    # drawing the dots.
    for name, group in groups:
        ax.scatter(group.iloc[:, 0], group.iloc[:, 1], group.iloc[:, 2],depthshade=False, s=3.0)
    # labeling the axis.
    ax.set_xlabel(X_3d.columns[0], size=size)
    ax.set_ylabel(X_3d.columns[1], size=size)
    ax.set_zlabel(X_3d.columns[2], size=size)
    fig.legend()
    return fig, ax

"""
Dimension reduction using MDS, not used in the final project.
"""
def MDS_and_plot(dist: pd.DataFrame, title="MDS_clustring", folder="plots", n_comp=3, labels=None):
    mds = MDS(n_components=n_comp, dissimilarity="precomputed")
    pos = mds.fit(dist).embedding_
    fig, ax = plot_data_3d(pd.DataFrame(pos), labels)
    ax.set_title(title)
    plt.savefig(f"{folder}/{title}.png")


"""
Dimension reduction using PCA and then plotting and saving the plot.
"""
def PCA_and_plot(df: pd.DataFrame, title="PCA_clustring", folder="plots", n_comp=3, labels=None):
    # doing dimension reduction.
    pca = PCA(n_components=n_comp)
    pos = pca.fit(df.T).components_

    # plotting and saving the plot.
    fig, ax = plot_data_3d(pd.DataFrame(pos).T, labels)
    ax.set_title(title)
    plt.savefig(f"{folder}/{title}.png")
