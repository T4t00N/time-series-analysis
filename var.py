import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.stattools import kpss, adfuller
from statsmodels.tools.sm_exceptions import InterpolationWarning

# --- Load Data ---
measurement = np.loadtxt('data/Measurement2.asc')
df = pd.DataFrame(measurement, columns=[f'ch{i+1}' for i in range(measurement.shape[1])])
print(f"Data loaded into DataFrame with {df.shape[1]} channels.\n")

# --- Stationarity Tests ---
def stationarity_tests(series, name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        k_stat, k_p, _, _ = kpss(series, regression="c", nlags="auto")
    a_stat, a_p, *_ = adfuller(series, autolag="AIC")
    print(f"{name}: KPSS stat={k_stat:.4f}, p={k_p:.4f} | ADF stat={a_stat:.4f}, p={a_p:.4f}")

# --- Run Tests ---
for col in df.columns:
    stationarity_tests(df[col], col)
