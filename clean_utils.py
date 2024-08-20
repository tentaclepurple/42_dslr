import pandas as pd
from ft_statistics import ft_median
import sys


def fill_missing_with_median(df: pd.DataFrame) -> pd.DataFrame:
    numerical_cols = df.select_dtypes(include=[float, int]).columns
    for col in numerical_cols:
        non_na_values = df[col].dropna().tolist()
        median_value = ft_median(non_na_values)
        if median_value is not None:
            df[col] = df[col].fillna(median_value)
    return df


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    numerical_cols = df.select_dtypes(include=[float, int]).columns

    for col in numerical_cols:
        min_value = df[col].min()
        max_value = df[col].max()
        df[col] = (df[col] - min_value) / (max_value - min_value)
    return df

def denormalize_data(normalized_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    numerical_cols = reference_df.select_dtypes(include=[float, int]).columns

    for col in numerical_cols:
        min_value = reference_df[col].min()
        max_value = reference_df[col].max()
        normalized_df[col] = normalized_df[col] * (max_value - min_value) + min_value
    
    return normalized_df



if __name__ == '__main__':
    
    try:
        path = 'datasets/dataset_train.csv'
                
        orig_df = pd.read_csv(path)
        
        print(orig_df)
        
        cleaned_df = fill_missing_with_median(orig_df.copy())
        
        print(cleaned_df)

        normalized_data = normalize_data(cleaned_df.copy())
        
        denormalized_data = denormalize_data(normalized_data.copy(), cleaned_df)
    
    except Exception as e:
        print(e)
        sys.exit(1)
