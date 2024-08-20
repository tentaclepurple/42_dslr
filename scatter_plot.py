import os
import pandas as pd
import matplotlib.pyplot as plt
import sys

from heatmap import plot_heatmap

house_colors = {
    'Gryffindor': 'red',
    'Hufflepuff': 'yellow',
    'Ravenclaw': 'blue',
    'Slytherin': 'green'
}

def plot_scatter(path: str):
    
    try:
        df = pd.read_csv(path)
    
    except FileNotFoundError:
        print('File not found')
        sys.exit(1)

    feature1 = 'Astronomy'
    feature2 = 'Defense Against the Dark Arts'
    house_feature = 'Hogwarts House'

    output_dir = "scatter_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 6))

    for house, color in house_colors.items():
        subset = df[df[house_feature] == house]
        plt.scatter(subset[feature1], subset[feature2], alpha=0.6, edgecolors='w', linewidth=0.5, label=house, color=color)

    plt.title(f'Scatter Plot of {feature1} vs {feature2}')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.grid(True)
    plt.legend(title="House")
    plt.savefig(f"{output_dir}/{feature1}_vs_{feature2}_scatter.png")


if __name__ == "__main__":
    
    try:
        path = sys.argv[1]

        plot_heatmap(path)

        plot_scatter(path)
        
    except IndexError:
        print('No file provided')
        sys.exit(1)
