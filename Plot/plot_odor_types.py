from matplotlib import pyplot as plt
import numpy as np

odor_names = ["almond", "anise", "creosote", "fishy", "foul", "musty", "none", "pungent", "spicy"]

def plot_odor_types(odor_tag, options=9, use_names=True, title="Distribution of odors", folder="plots"):
    odors=range(options)
    if use_names:
        odors = odor_names
    weighted = np.zeros(options)
    for i in range(options):
        weighted[i] = list(odor_tag).count(i)
    fig, ax = plt.subplots()
    ax.bar(height=weighted, x=odors, color='b')
    ax.set_title(title)
    ax.set_xlabel("Odors", color='r')
    ax.set_ylabel("Amount", color='r')
    ax.grid(axis='y')
    plt.savefig(f"{folder}/{title}")
