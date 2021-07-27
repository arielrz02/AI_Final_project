import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_model_pref(folder="plots", name="Model preformance", bf=False):
    df = pd.DataFrame([[0.98, "full", "NN, micro"], [0.83, "full", "RF, micro"],
                       [0.64, "missing", "NN, micro"], [0.5, "missing", "RF, micro"],
                       [0.59, "full", "NN, macro"], [0.5, "full", "RF, macro"],
                       [0.51, "missing", "NN, macro"], [0.3, "missing", "RF, macro"]
                       ],
                      columns=["F1 score", "Data", "Model and type"])
    df_better_feat = pd.DataFrame([[0.98, "full", "NN, micro"], [0.82, "full", "RF, micro"],
                       [0.62, "missing", "NN, micro"], [0.51, "missing", "RF, micro"],
                       [0.63, "full", "NN, macro"], [0.53, "full", "RF, macro"],
                       [0.51, "missing", "NN, macro"], [0.3, "missing", "RF, macro"]
                       ],
                      columns=["F1 score", "Data", "Model and type"])
    df_dict = {True: df_better_feat, False: df}
    fig, ax = plt.subplots()
    ax = sns.barplot(data=df_dict[bf], y="F1 score", x="Model and type", hue="Data", palette="mako")
    ax.set_xlabel(color='r', xlabel="Model and type")
    ax.set_ylabel(color='r', ylabel="F1 score")
    ax.set_title(name)
    plt.savefig(f"{folder}/{name}.png")

if __name__ == "__main__":
    plot_model_pref(bf=True, name="Better features model preformance")
    plot_model_pref()
