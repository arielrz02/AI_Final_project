import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def plot_data_3d(X_3d: pd.DataFrame, color: pd.Series, color_dict=None, labels_dict: dict = None, size=10):
    fig = plt.figure()
    ax = Axes3D(fig)
    groups = X_3d.groupby(color)
    for name, group in groups:
        if color_dict is None and labels_dict is None:
            ax.scatter(group.iloc[:, 0], group.iloc[:, 1], group.iloc[:, 2],depthshade=False)
        elif color_dict is None:
            ax.scatter(group.iloc[:, 0], group.iloc[:, 1], group.iloc[:, 2], label=labels_dict[name],depthshade=False)
        elif labels_dict is None:
            ax.scatter(group.iloc[:, 0], group.iloc[:, 1], group.iloc[:, 2], c=color_dict[name],depthshade=False)
        else:
            ax.scatter(group.iloc[:, 0], group.iloc[:, 1], group.iloc[:, 2],
                       c=color_dict[name], label=labels_dict[name],depthshade=False)
    ax.set_xlabel(X_3d.columns[0], size=size)
    ax.set_ylabel(X_3d.columns[1], size=size)
    ax.set_zlabel(X_3d.columns[2], size=size)
    fig.legend()
    return fig, ax


def MDS_and_plot(dist: pd.DataFrame, title="mushrooms_clustring", folder="Plot", n_comp=3, labels=pd.Series(range(256))):
    mds = MDS(n_components=n_comp, dissimilarity="precomputed")
    pos = mds.fit(dist).embedding_
    fig, ax = plot_data_3d(pd.DataFrame(pos), labels)
    ax.set_title(title)
    plt.savefig(f"{title}.png")