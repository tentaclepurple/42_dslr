import numpy as np
import pandas as pd
import sys
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from clean_utils import fill_missing_with_median, normalize_data

def sigmoid(x):
    '''
    The sigmoid function takes a number x and transforms it into a value between 0 and 1.
This is done through the formula 1 / (1 + e^-x), where e is the base of the natural logarithm.
When x is very large, e^-x approaches 0, so the output of the sigmoid function approaches 1.
When x is very small, e^-x approaches infinity, so the output of the sigmoid function approaches 0.
When x is 0, e^-x is 1, so the output of the sigmoid function is 0.5.
Therefore, the sigmoid function maps positive numbers to values ​​between 0.5 and 1,
and maps negative numbers to values ​​between 0 and 0.5.
    '''
    return 1 / (1 + np.exp(-x)) 


def cost_function(x, y, theta):
    m = len(y)
    h = sigmoid(x @ theta)
    cost = -1 / m * (y @ np.log(h) + (1 - y) @ np.log(1 - h))
    return cost


def mini_batch_gradient_descent(x, y, theta, alpha, iterations, batch_size=32):
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        # Mezclar los datos para evitar patrones en los mini-batches
        indices = np.random.permutation(m)
        x_shuffled = x[indices]
        y_shuffled = y[indices]
        
        # Procesar mini-batches
        for start in range(0, m, batch_size):
            end = start + batch_size
            x_mini = x_shuffled[start:end]
            y_mini = y_shuffled[start:end]

            # Calcular la predicción para el mini-batch
            h = sigmoid(x_mini @ theta)

            # Actualizar los parámetros con el gradiente del mini-batch
            theta = theta - (alpha / batch_size) * x_mini.T @ (h - y_mini)

        # Almacenar el costo después de cada iteración (época)
        cost_history[i] = cost_function(x, y, theta)
    
    return theta, cost_history


def one_vs_all(x, y, classes, alpha, iterations):
    thetas = np.zeros((len(classes), x.shape[1]))
    cost_histories = []
    for i, c in tqdm(enumerate(classes)):
        binary_y = np.where(y == c, 1, 0)
        theta, cost_history = mini_batch_gradient_descent(x, binary_y, thetas[i], alpha, iterations)
        thetas[i] = theta
        cost_histories.append(cost_history)
    return thetas, cost_histories


def logistic_regression(df):
    
    category = df['Hogwarts House'].astype('category')
    y = category.cat.codes.to_numpy()
    classes = category.cat.categories.tolist()

    df = df[['Ancient Runes', 'Defense Against the Dark Arts', 'Charms', 'Divination']]
    x = df.to_numpy()
    x = np.insert(x, 0, 1, axis=1) # Add bias term

    alpha = 0.01
    iterations = 10000
    
    thetas, cost_histories = one_vs_all(x, y, np.arange(len(classes)), alpha, iterations)

    output_dir = 'weights'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'logreg_weigths_mbgd.pkl')

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
    plt.savefig('weights/cost_mbgd.png')


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