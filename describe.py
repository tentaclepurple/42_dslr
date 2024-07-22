import pandas as pd
from statistics import ft_mean, ft_median, ft_std, ft_sort, ft_quartile


def ft_describe(df: pd.DataFrame) -> pd.DataFrame:
    stats = {}

    for column in df.select_dtypes(include=[float, int]).columns:
        col_stats = {}
        column_data = df[column].dropna().to_list()

        if column_data:
            col_stats["count"] = len(column_data)
            col_stats["mean"] = ft_mean(column_data)
            col_stats["std"] = ft_std(column_data)
            col_stats["min"] = ft_sort(column_data)[0]
            col_stats["25%"] = ft_quartile(column_data)[0]
            col_stats["50%"] = ft_median(column_data)
            col_stats["75%"] = ft_quartile(column_data)[1]
            col_stats["max"] = ft_sort(column_data)[-1]
        else:
            col_stats["count"] = 0


        stats[column] = col_stats
    
    dataf = pd.DataFrame(stats)

    print(dataf)
    return dataf


def main():
    path = "datasets/dataset_test.csv"
    df = pd.read_csv(path)

    print("PANDAS HEAD:")
    print(df)

    print("PANDAS DESCRIBE:")
    print(df.describe())

    ft_describe(df)


if __name__ == "__main__":
    main()