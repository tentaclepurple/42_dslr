import numpy as np
import pandas as pd
import pickle
import sys
import os
from clean_utils import fill_missing_with_median, normalize_data

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_one_vs_all(x, thetas):
    probabilities = sigmoid(x @ thetas.T)
    return np.argmax(probabilities, axis=1)

def predict(df, thetas) -> None:
    df = df[['Ancient Runes', 'Defense Against the Dark Arts', 'Charms', 'Divination']]
    x = df.to_numpy()
    x = np.insert(x, 0, 1, axis=1)

    if x.shape[1] != thetas.shape[1]:
        print(f"Dimension mismatch: x has {x.shape[1]} columns but thetas have {thetas.shape[1]} columns.")
        sys.exit(1)
        
    predictions = predict_one_vs_all(x, thetas)
    house_names = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    predictions_houses = [house_names[pred] for pred in predictions]

    results = pd.DataFrame({
        'Index': df.index,
        'Hogwarts House': predictions_houses
    })

    output_file = 'houses.csv'
    results.to_csv(output_file, index=False)

if __name__ == '__main__':
    try:
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
        
    except Exception as e:
        print(e)
        sys.exit(1)
