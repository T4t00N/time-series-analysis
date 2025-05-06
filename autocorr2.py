import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load data
measurement1 = np.loadtxt('data/Measurement2.asc')


# Define the data used for exploration
data = measurement1
# Define channel names
Accelerometer = [f'ch{i+1}' for i in range(10)]
#Save it to dataframe
df = pd.DataFrame(data, columns=Accelerometer)

def autocorr_pandas(series, lag=1):
    """
    Calculates the autocorrelation of a Pandas Series at a specified lag.

    Args:
        series (pd.Series): The time series data.
        lag (int): The lag (number of time steps) to consider.

    Returns:
        float: The autocorrelation coefficient at the specified lag.
               Returns NaN if the series has fewer than lag + 1 elements.
    """
    return series.autocorr(lag=lag)

def autocorrelogram(series, max_lag):
    """
    Calculates the autocorrelation for a range of lags.

    Args:
        series (pd.Series or np.ndarray): The time series data.
        max_lag (int): The maximum lag to calculate autocorrelation for.

    Returns:
        np.ndarray: An array of autocorrelation coefficients for lags from 0 to max_lag.
    """
    if isinstance(series, pd.Series):
        return np.array([series.autocorr(lag=i) for i in range(max_lag + 1)])
    elif isinstance(series, np.ndarray):
        n = len(series)
        mean = np.mean(series)
        denominator = np.sum((series - mean)**2)
        if denominator == 0:
            return np.full(max_lag + 1, np.nan)
        return np.array([np.sum((series[:n-i] - mean) * (series[i:] - mean)) / denominator if n > i else np.nan for i in range(max_lag + 1)])
    else:
        raise ValueError("Input must be a Pandas Series or a NumPy array.")

# Calculate and visualize the autocorrelogram for each channel
max_lags = 100  # You can adjust this number

for channel in df.columns:
    autocorr_values = autocorrelogram(df[channel], max_lags)

    plt.figure(figsize=(12, 6))
    plt.stem(range(max_lags + 1), autocorr_values)
    plt.title(f"Autocorrelogram of {channel}")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation Coefficient")
    plt.grid(True)
    plt.ylim(-1, 1)  # Set y-axis limits to -1 and 1
    plt.savefig(f'autocorr_{channel}.png')  # Save each plot as a PNG file
    plt.show()