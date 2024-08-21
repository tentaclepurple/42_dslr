import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from clean_utils import fill_missing_with_median, normalize_data

def sigmoid(x):
    '''
    La función sigmoide toma un número x y lo transforma en un valor entre 0 y 1.
    Esto se hace a través de la fórmula 1 / (1 + e^-x), donde e es la base del logaritmo natural.
    Cuando x es muy grande, e^-x se acerca a 0, por lo que la salida de la función sigmoide se acerca a 1.
    Cuando x es muy pequeño, e^-x se acerca a infinito, por lo que la salida de la función sigmoide se acerca a 0.
    Cuando x es 0, e^-x es 1, por lo que la salida de la función sigmoide es 0.5.
    Por lo tanto, la función sigmoide mapea los números positivos a valores entre 0.5 y 1,
    y mapea los números negativos a valores entre 0 y 0.5.
    '''
    return 1 / (1 + np.exp(-x)) 

def cost_function(x, y, theta):
    m = len(y)
    h = sigmoid(x @ theta)
    cost = -1 / m * (y @ np.log(h) + (1 - y) @ np.log(1 - h))
    return cost

def gradient_descent(x, y, theta, alpha, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        '''
        Aquí, x @ theta calcula una combinación lineal de las características en x y los pesos en theta.
        Esto incluye el término de sesgo, porque x se supone que tiene una columna de unos al principio,
        y theta[0] es el peso para esta columna de unos.
        '''
        h = sigmoid(x @ theta)    
        '''
        Aquí, actualizamos todos los pesos en theta, incluyendo el término de sesgo.
        La actualización se hace de tal manera que el error (h - y) se distribuye a cada peso
        en proporción a la contribución de la característica correspondiente a la predicción.
        '''  
        theta = theta - (alpha / m) * x.T @ (h - y)
        cost_history[i] = cost_function(x, y, theta)
    return theta, cost_history

def one_vs_all(x, y, classes, alpha, iterations):
    thetas = np.zeros((len(classes), x.shape[1]))
    cost_histories = []
    for i, c in enumerate(classes):
        binary_y = np.where(y == c, 1, 0)
        theta, cost_history = gradient_descent(x, binary_y, thetas[i], alpha, iterations)
        thetas[i] = theta
        cost_histories.append(cost_history)
    return thetas, cost_histories

def logistic_regression(df):
    
    category = df['Hogwarts House'].astype('category')
    y = category.cat.codes.to_numpy()
    classes = category.cat.categories.tolist()
    print('Classes: ', classes)

    df = df[['Ancient Runes', 'Defense Against the Dark Arts', 'Charms', 'Divination', 'Potions']]
    x = df.to_numpy()
    x = np.insert(x, 0, 1, axis=1) # Add bias term

    alpha = 0.01
    iterations = 10000
    
    thetas, cost_histories = one_vs_all(x, y, np.arange(len(classes)), alpha, iterations)

    output_dir = 'weights'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'logreg_weigths.pkl')

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
    plt.savefig('weights/cost.png')


if __name__ == "__main__":
    path = 'datasets/dataset_train.csv'
    df_original = pd.read_csv(path)
    df_clean = fill_missing_with_median(df_original)
    df_normalized = normalize_data(df_clean)
    logistic_regression(df_normalized)
