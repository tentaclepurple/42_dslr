import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_all_scatters_in_one(path: str):
    df = pd.read_csv(path)

    # Seleccionar solo las columnas numéricas
    numerical_df = df.select_dtypes(include=[float, int])

    # Crear el directorio si no existe
    output_dir = "scatter_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Crear un Pairplot
    pairplot = sns.pairplot(numerical_df)
    
    # Guardar el gráfico
    pairplot.savefig(f"{output_dir}/pairplot_all_features.png")


if __name__ == "__main__":
    path = "datasets/dataset_train.csv"
    plot_all_scatters_in_one(path)
