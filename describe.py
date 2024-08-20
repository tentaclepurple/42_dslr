import pandas as pd
from ft_statistics import ft_mean, ft_median, ft_std, ft_sort, ft_quartile


def ft_kurtosis(args):
    """
    Kurtosis measures the "tailedness" of the probability distribution 
    of a real-valued random variable. It provides an indication of 
    the extremity of outliers and the shape of the tails of the distribution. 
    
    Specifically:
    
    Positive kurtosis indicates that the distribution has heavier tails 
        and a sharper peak than the normal distribution. 
        This means there are more outliers than expected.
    Negative kurtosis indicates that the distribution has 
        lighter tails and a flatter peak than the normal distribution.
        This means there are fewer outliers than expected.
    Zero kurtosis (when the excess kurtosis is zero) 
        indicates that the distribution has a shape similar to the 
        normal distribution, with tails and peaks comparable to those 
        of a normal distribution.
    """
    n = len(args)
    if n < 4:
        return None  # Kurtosis is not defined for less than 4 data points
    
    mean = ft_mean(args)
    std_dev = ft_std(args)
    
    if std_dev == 0:
        return None
    
    fourth_moment = sum(((x - mean) / std_dev) ** 4 for x in args) / n
    
    kurt = (n * (n + 1) * fourth_moment - 3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    
    return kurt


def ft_var(args):
    """Calculate the variance of the given list of numbers."""
    if len(args) == 0:
        return None
    mean = ft_mean(args)
    variance = sum((arg - mean) ** 2 for arg in args) / (len(args) - 1)
    return variance


def ft_sem(args):
    """Calculate the standard error of the mean of the given list of numbers."""
    if len(args) < 2:
        return None
    std_dev = ft_std(args)
    return std_dev / (len(args) ** 0.5)


def ft_describe(df: pd.DataFrame) -> pd.DataFrame:
    stats = {}

    for column in df.select_dtypes(include=[float, int]).columns:
        col_stats = {}
        column_data = df[column].dropna().to_list()

        if column_data:
            col_stats["count"] = len(column_data)
            col_stats["mean"] = ft_mean(column_data)
            col_stats["std"] = ft_std(column_data)
            col_stats["variance"] = ft_var(column_data)
            col_stats["sem"] = ft_sem(column_data)
            col_stats["min"] = ft_sort(column_data)[0]
            col_stats["25%"] = ft_quartile(column_data)[0]
            col_stats["50%"] = ft_median(column_data)
            col_stats["75%"] = ft_quartile(column_data)[1]
            col_stats["max"] = ft_sort(column_data)[-1]
            col_stats["IQR"] = col_stats["75%"] - col_stats["25%"]
            col_stats["kurtosis"] = ft_kurtosis(column_data)

        else:
            col_stats["count"] = 0


        stats[column] = col_stats
    
    dataf = pd.DataFrame(stats)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    print(dataf)
    return dataf


def main():
    path = "datasets/dataset_train.csv"
    df = pd.read_csv(path)

    print("PANDAS HEAD:")
    print(df)

    print("PANDAS DESCRIBE:")
    print(df.describe())

    ft_describe(df)


if __name__ == "__main__":
    main()