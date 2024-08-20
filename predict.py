import sys
import os
import pandas as pd
import pickle
from clean_utils import fill_missing_with_median

def predict(df):
    """
    Predict the target variable of the dataset
    :param df: the dataset
    :return: the predicted target variable
    """
    cleaned_df = fill_missing_with_median(df)




if __name__ == "__main__":
    try:
        path = sys.argv[1]

        #path = 'datasets/dataset_test.csv'

        weights_path = sys.argv[2]
        if os.path.exists(weights_path):
            if os.access(weights_path, os.R_OK):
                with open(weights_path, 'rb') as file:
                    weights = pickle.load(file)

    
        df = pd.read_csv(path)
        
        predict(df)
            
    except Exception as e:
        print(e)
        sys.exit(1)