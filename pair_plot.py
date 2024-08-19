import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

def plot_all_scatters_in_one(path: str):
    
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print('File not found')
        sys.exit(1)

    numerical_df = df.select_dtypes(include=[float, int])
    
    numerical_df['Hogwarts House'] = df['Hogwarts House']

    output_dir = "pair_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pairplot = sns.pairplot(numerical_df, hue='Hogwarts House', palette={
        'Gryffindor': 'red',
        'Hufflepuff': 'yellow',
        'Ravenclaw': 'blue',
        'Slytherin': 'green'
    })

    pairplot.savefig(f"{output_dir}/pairplot_all_features.png")


if __name__ == "__main__":
    
    try:
        path = sys.argv[1]

        plot_all_scatters_in_one(path)
        
    except IndexError:
        print('No file provided')
        sys.exit(1)
