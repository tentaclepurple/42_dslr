import pandas as pd
import sys


def fill_missing_with_mean(df: pd.DataFrame) -> pd.DataFrame:
    numerical_cols = df.select_dtypes(include=[float, int]).columns
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
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
        
        cleaned_df = fill_missing_with_mean(orig_df.copy())

        normalized_data = normalize_data(cleaned_df.copy())
        
        denormalized_data = denormalize_data(normalized_data.copy(), cleaned_df)
    
    except Exception as e:
        print(e)
        sys.exit(1)
