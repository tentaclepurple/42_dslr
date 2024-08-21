import numpy as np
import pandas as pd
import sys
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from clean_utils import fill_missing_with_median, normalize_data


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Para estabilidad numérica
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cost_function_softmax(x, y, theta):
    m = len(y)
    z = x @ theta
    h = softmax(z)
    # Crear un vector de 1-hot encoding para y
    y_one_hot = np.eye(h.shape[1])[y]
    cost = -np.sum(y_one_hot * np.log(h)) / m
    return cost


def gradient_descent(x, y, theta, alpha, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    y_one_hot = np.eye(theta.shape[1])[y]
    for i in tqdm(range(iterations)):
        z = x @ theta
        h = softmax(z)    

        theta = theta - (alpha / m) * x.T @ (h - y_one_hot)
        cost_history[i] = cost_function_softmax(x, y, theta)
    return theta, cost_history


def logistic_regression(df):
    
    category = df['Hogwarts House'].astype('category')
    y = category.cat.codes.to_numpy()
    classes = category.cat.categories.tolist()

    df = df[['Ancient Runes', 'Defense Against the Dark Arts', 'Charms', 'Divination']]
    x = df.to_numpy()
    x = np.insert(x, 0, 1, axis=1) # Add bias term

    alpha = 0.01
    iterations = 10000
    
    theta = np.zeros((x.shape[1], (len(classes))))

    
    thetas, cost_histories = gradient_descent(x, y, theta, alpha, iterations)

    output_dir = 'weights'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'logreg_weigths_softmax.pkl')

    with open(output_path, 'wb') as f:
        pickle.dump(thetas, f)
    print(f'Weights saved in "{output_path}".')

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()
    for i, (c, cost_history) in enumerate(zip(classes, cost_histories)):
        axs[i].plot(cost_history) 
        axs[i].set_title(c)
        axs[i].set_xlabel('Iterations')
        axs[i].set_ylabel('Cost')
    plt.tight_layout()
    plt.savefig('weights/cost_softmax.png')


if __name__ == "__main__":
    try:
        path = sys.argv[1]
        
        df_original = pd.read_csv(path)
        df_clean = fill_missing_with_median(df_original)
        df_normalized = normalize_data(df_clean)
        logistic_regression(df_normalized)
        
    except Exception as e:
        print(e)
        sys.exit(1)
