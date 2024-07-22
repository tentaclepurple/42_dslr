import pandas as pd
from statistics import ft_mean, ft_median, ft_std, ft_sort, ft_quartile 


class Describe():
    """
    Class to describe a dataset like pandas.describe()
    """
    def __init__(self, df: pd.DataFrame):
        """
        creates a Describe df object
        """
        #self.column_names = df.columns.to_list()
        #self.describe_df = pd.DataFrame(columns=self.column_names)
        self.describe_df = self.create_df(df)
    
    def create_df(df):
        stats = {}

        for column in df.columuns:
                    



def ft_describe(df):
    Desc = Describe(df)
    print(Desc.describe_df)

    



def main():
    path = "datasets/dataset_test.csv"
    df = pd.read_csv(path)

    #print(df)

    print("PANDAS DESCRIBE:")
    print(df.describe())

    ft_describe(df)


if __name__ == "__main__":
    main()