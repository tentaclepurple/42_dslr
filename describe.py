import pandas as pd


def ft_describe(path):

    df = pd.read_csv(path)
    print(df)



def main():
    path = "datasets/dataset_test.csv"
    ft_describe(path)


if __name__ == "__main__":
    main()