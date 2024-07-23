import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_all_scatters_in_one(path: str):
    df = pd.read_csv(path)

    numerical_df = df.select_dtypes(include=[float, int])

    output_dir = "pair_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pairplot = sns.pairplot(numerical_df)
    
    pairplot.savefig(f"{output_dir}/pairplot_all_features.png")


if __name__ == "__main__":
    path = "datasets/dataset_train.csv"
    plot_all_scatters_in_one(path)
