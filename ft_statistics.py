from typing import Any


def ft_mean(args):
    """
    take in *args a quantity of unknown number and return the mean.
    """
    if len(args) == 0:
        return None
    return sum(args) / len(args)


def ft_median(args: list):
    """
    take in *args a quantity of unknown number and return the median.
    """
    
    if not args:
        return None

    args = ft_sort(args)
    
    if len(args) % 2 == 0:
        return (args[len(args) // 2] + args[len(args) // 2 - 1]) / 2
    else:
        return args[len(args) // 2]


def ft_quartile(args: list):
    """
    take in *args a quantity of unknown number and return 25% and 75% quartile.
    """
    args = sorted(args)
    n = len(args)
    
    def percentile(p):
        k = p * (n - 1)
        f = int(k)
        c = k - f
        if f + 1 < n:
            return args[f] + (args[f + 1] - args[f]) * c
        else:
            return args[f]

    q1 = percentile(0.25)
    q3 = percentile(0.75)
    return [q1, q3]


def ft_std(args):
    """
    take in *args a quantity of unknown number
    and return the standard deviation.
    """
    if len(args) == 0:
        return None
    mean = ft_mean(args)
    variance = sum((arg - mean) ** 2 for arg in args) / (len(args) - 1)
    return variance ** 0.5


def ft_sort(args):
    """
    take in *args a quantity of unknown number and return a sorted list.
    """
    if len(args) == 0:
        return []
    args = list(args)
    for i in range(len(args)):
        for j in range(len(args)):
            if args[i] < args[j]:
                args[i], args[j] = args[j], args[i]
    return args


def ft_statistics(*args: Any, **kwargs: Any) -> None:
    """
    take in *args a quantity of unknown number and make
    the Mean, Median, Quartile (25% and 75%), Standard Deviation
    and Variance according to the **kwargs
    ask.
    """
    try:
        if len(args) == 0:
            for value in kwargs.items():
                print("ERROR")
            return
        for arg in args:
            if not isinstance(arg, (int, float)):
                raise AssertionError("ERROR")
        sorted_args = ft_sort(*args)
        for key, value in kwargs.items():
            if value == "mean":
                print(f"mean: {ft_mean(*args)}")
            elif value == "median":
                print(f"median: {ft_median(sorted_args)}")
            elif value == "quartile":
                print(f"quartile: {ft_quartile(sorted_args)}")
            elif value == "std":
                if len(args) < 2:
                    print("Std error")
                    continue
                print(f"std: {ft_std(*args)}")
            elif value == "var":
                if len(args) < 2:
                    print("Var error")
                    continue
                print(f"var: {ft_std(*args) ** 2}")

    except AssertionError as e:
        print(e)
        return None


def main():
    ft_statistics(1, 42, 360, 11, 64, too="mean", pe="median", jo="quartile")
    print("-----")
    ft_statistics(5, 75, 450, 18, 597, 27474, 48575, hello="std", world="var")
    print("-----")
    ft_statistics(5, 75, 450, 18, 597, 27474, 48575, ehhe="eh", eejn="kd")
    print("-----")
    ft_statistics(toto="mean", tutu="median", tata="quartile")
    print("-----")
    ft_statistics(12, toto="var", tutu="mean", tata="quartile")


if __name__ == "__main__":
    main()
