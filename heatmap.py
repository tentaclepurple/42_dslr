import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(path: str):
    df = pd.read_csv(path)

    # Seleccionar solo las columnas numéricas
    numerical_df = df.select_dtypes(include=[float, int])

    # Calcular la matriz de correlación
    corr_matrix = numerical_df.corr()

    # Crear el directorio si no existe
    output_dir = "heatmaps"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Crear el mapa de calor
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Heatmap of Feature Correlations')
    plt.savefig(f"{output_dir}/correlation_heatmap.png")


if __name__ == "__main__":
    path = "datasets/dataset_train.csv"
    plot_heatmap(path)
