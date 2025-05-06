import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.stattools import kpss  # Import KPSS test
from statsmodels.tools.sm_exceptions import InterpolationWarning

# --- Load Data ---
measurement1 = np.loadtxt('data/Measurement2.asc')
num_channels = measurement1.shape[1]
ch_names = [f'ch{i + 1}' for i in range(num_channels)]
df = pd.DataFrame(measurement1, columns=ch_names)
print(f"Data loaded into DataFrame with {num_channels} channels.\n")

# --- KPSS Test Function ---

def kpss_test(timeseries, channel_name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)  # Suppress KPSS warning
        stat, p_value, lags, crit = kpss(timeseries, regression="c", nlags="auto")
    print(f"Channel {channel_name} - KPSS Test Statistic: {stat:.4f}, p-value: {p_value:.4f}, lags: {lags}")


# --- Run KPSS on all channels ---
for col in df.columns:
    kpss_test(df[col], col)
