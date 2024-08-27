import numpy as np
import pandas as pd
import sys
import os
import pickle
from tqdm import tqdm
from clean_utils import fill_missing_with_median, normalize_data


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Para estabilidad numérica
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def predict_softmax(x, theta):

    if theta.shape == (4, 5):
        theta = theta.T
    elif theta.shape != (5, 4):
        raise ValueError(f"Expected theta to have shape (5, 4), but got {theta.shape}")
    
    if x.shape[1] == theta.shape[0] - 1:
        x = np.insert(x, 0, 1, axis=1)
    
    z = x @ theta
    h = softmax(z)
    predictions = np.argmax(h, axis=1)
    return predictions


def predict(df, thetas) -> None:
    df = df[['Ancient Runes', 'Defense Against the Dark Arts', 'Charms', 'Divination']]
    x = df.to_numpy()

    predictions = predict_softmax(x, thetas)
    house_names = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    predictions_houses = [house_names[pred] for pred in predictions]

    results = pd.DataFrame({
        'Index': df.index,
        'Hogwarts House': predictions_houses
    })

    output_file = 'houses.csv'
    results.to_csv(output_file, index=False)


if __name__ == '__main__':
        path = sys.argv[1]
        df = pd.read_csv(path)
        df_original = pd.read_csv(path)
        df_clean = fill_missing_with_median(df_original)
        df_normalized = normalize_data(df_clean)
        
        weights_path = sys.argv[2]
        if os.path.exists(weights_path):
            if os.access(weights_path, os.R_OK):
                with open(weights_path, 'rb') as file:
                    weights = pickle.load(file)
        
        predict(df_normalized, weights)
