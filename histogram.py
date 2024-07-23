import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def plot_histograms(path: str):
    df = pd.read_csv(path)

    houses = df["Hogwarts House"].unique()
    courses = ['Arithmancy', 'Astronomy', 'Herbology', 
        'Defense Against the Dark Arts', 'Divination',
        'Muggle Studies', 'Ancient Runes', 'History of Magic',
        'Transfiguration', 'Potions', 'Care of Magical Creatures',
        'Charms', 'Flying']

    output_dir = "histograms"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for course in tqdm(courses):
        plt.figure(figsize=(10, 6))
        for house in houses:
            scores = df[df['Hogwarts House'] == house][course].dropna()
            plt.hist(scores, alpha=0.5, label=house, bins=20)
        plt.title(f'Score Distribution for {course}')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.savefig(f"{output_dir}/{course}_histogram.png")
        plt.close()


if __name__ == "__main__":
    path = "datasets/dataset_train.csv"
    plot_histograms(path)
