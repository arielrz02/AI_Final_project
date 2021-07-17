import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_model_pref(folder="plots", name="Model preformance"):
    df = pd.DataFrame([[0.98, "full", "NN, micro"], [0.83, "full", "RF, micro"],
                       [0.64, "missing", "NN, micro"], [0.5, "missing", "RF, micro"],
                       [0.59, "full", "NN, macro"], [0.5, "full", "RF, macro"],
                       [0.51, "missing", "NN, macro"], [0.3, "missing", "RF, macro"]
                       ],
                      columns=["F1 score", "Data", "Model and type"])
    fig, ax = plt.subplots()
    ax = sns.barplot(data=df, y="F1 score", x="Model and type", hue="Data", palette="mako")
    ax.set_xlabel(color='r', xlabel="Model and type")
    ax.set_ylabel(color='r', ylabel="F1 score")
    plt.savefig(f"{folder}/{name}.png")
    plt.show()

plot_model_pref()