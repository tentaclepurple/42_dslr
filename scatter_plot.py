import os
import pandas as pd
import matplotlib.pyplot as plt
from heatmap import plot_heatmap


def plot_scatter(path: str):
    df = pd.read_csv(path)

    feature1 = 'Transfiguration'
    feature2 = 'History of Magic'

    output_dir = "scatter_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 6))
    plt.scatter(df[feature1], df[feature2], alpha=0.5, edgecolors='w', linewidth=0.5)
    plt.title(f'Scatter Plot of {feature1} vs {feature2}')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.grid(True)
    plt.savefig(f"{output_dir}/{feature1}_vs_{feature2}_scatter.png")


if __name__ == "__main__":
    path = "datasets/dataset_train.csv"

    plot_heatmap(path)

    plot_scatter(path)
