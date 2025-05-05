import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load data
measurement1 = np.loadtxt('data/Measurement1.asc')
measurement2 = np.loadtxt('data/Measurement2.asc')

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



for channel in df.columns:
    for lag in range(1, 6):  # Calculate autocorrelation for lags 1 to 5
        autocorr_value = autocorr_pandas(df[channel], lag=lag)
        print(f"Autocorrelation of {channel} at lag {lag}: {autocorr_value}")

