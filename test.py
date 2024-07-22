import pandas as pd
import numpy as np

def mi_describe(df):
    # Inicializamos un diccionario para almacenar las estadísticas
    stats = {}
    
    for column in df.columns:
        col_stats = {}
        
        # Calculamos las estadísticas solo para columnas numéricas
        if pd.api.types.is_numeric_dtype(df[column]):
            col_stats['count'] = df[column].count()
            col_stats['mean'] = df[column].mean()
            col_stats['std'] = df[column].std()
            col_stats['min'] = df[column].min()
            col_stats['25%'] = df[column].quantile(0.25)
            col_stats['50%'] = df[column].quantile(0.50)
            col_stats['75%'] = df[column].quantile(0.75)
            col_stats['max'] = df[column].max()
        else:
            # Para columnas no numéricas, solo contamos los valores únicos y la frecuencia del más común
            col_stats['count'] = df[column].count()
            col_stats['unique'] = df[column].nunique()
            col_stats['top'] = df[column].mode().iloc[0]
            col_stats['freq'] = df[column].value_counts().iloc[0]
        
        stats[column] = col_stats
    
    # Creamos un nuevo DataFrame con las estadísticas
    return pd.DataFrame(stats)

# Ejemplo de uso
df = pd.read_csv("datasets/dataset_test.csv")

resultado = mi_describe(df)
print(resultado)