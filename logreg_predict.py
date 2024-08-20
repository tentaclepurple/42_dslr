import numpy as np
import pandas as pd
import pickle
import sys
from clean_utils import fill_missing_with_mean, normalize_data

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_one_vs_all(x, thetas):
    probabilities = sigmoid(x @ thetas.T)
    return np.argmax(probabilities, axis=1)

def predict(df, thetas) -> None:
    df = df.select_dtypes(include=[float, int])
    x = df.to_numpy()

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
        path = 'datasets/dataset_test.csv'
        df = pd.read_csv(path)
        with open('weights/logreg_weigths.pkl', 'rb') as f:
            thetas = pickle.load(f)
    except Exception as e:
        print(e)
        sys.exit(1)
    predict(df, thetas)