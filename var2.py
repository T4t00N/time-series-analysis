import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import kpss  # Import KPSS test
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
from scipy.signal import detrend

warnings.filterwarnings("ignore")

def check_stationarity(df, significance_level=0.05):
    """
    Checks the stationarity of each column in a Pandas DataFrame using the
    Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test.

    Args:
        df (pd.DataFrame): The input DataFrame.
        significance_level (float): The significance level for the tests. Default is 0.05.

    Returns:
        pd.DataFrame: A DataFrame with the KPSS test results for each column.
        bool: True if all series are stationary according to the KPSS test, False otherwise.
    """
    results = {}
    for col in df.columns:
        # KPSS Test
        kpss_result = kpss(df[col], regression='c', nlags='auto')  # Use 'c' for constant trend
        kpss_stationary = kpss_result[1] > significance_level  # KPSS null is stationarity

        results[col] = {
            'KPSS Statistic': kpss_result[0],
            'KPSS p-value': kpss_result[1],
            'KPSS Conclusion': 'Stationary' if kpss_stationary else 'Non-Stationary',
        }
    results_df = pd.DataFrame(results).T  # Transpose for better readability
    all_stationary = all(results_df['KPSS Conclusion'] == 'Stationary')
    return results_df, all_stationary



def scale_dataframe(df, method='standardize'):
    """
    Scales the columns of a Pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        method (str): The scaling method.  'standardize' (Z-score) or 'minmax'.
            Default is 'standardize'.

    Returns:
        pd.DataFrame: The scaled DataFrame.
    """
    df_scaled = df.copy()
    if method == 'standardize':
        for col in df_scaled.columns:
            df_scaled[col] = (df_scaled[col].values - df_scaled[col].mean()) / df_scaled[col].std()
    elif method == 'minmax':
        for col in df_scaled.columns:
            df_scaled[col] = (df_scaled[col].values - df_scaled[col].min()) / (df_scaled[col].max() - df_scaled[col].min())
    else:
        raise ValueError("Invalid scaling method. Choose 'standardize' or 'minmax'.")
    return df_scaled

def apply_moving_average(df, window=5):
    """
    Applies a moving average filter to the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        window (int): The window size for the moving average.  Default is 5.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    df_ma = df.copy()
    for col in df_ma.columns:
        df_ma[col] = df_ma[col].rolling(window=window).mean()
    return df_ma.dropna()

def detrend_dataframe(df):
    """
   Detrends the data in the DataFrame using linear detrending.

   Args:
       df (pd.DataFrame): The input DataFrame

   Returns:
       pd.DataFrame: The detrended DataFrame
   """
    df_detrended = df.copy()
    for col in df_detrended.columns:
        df_detrended[col] = detrend(df_detrended[col])
    return df_detrended

def smooth_dataframe(df, window_size=500):
    """
    Applies a simple moving average smoothing to the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        window_size (int): The window size for the moving average. Default is 500.

    Returns:
        pd.DataFrame: The smoothed DataFrame.
    """
    df_smoothed = df.copy()
    for col in df_smoothed.columns:
        df_smoothed[col] = df_smoothed[col].rolling(window=window_size).mean()
    return df_smoothed.dropna()

# --- 0. Load Data (using your existing approach) ---
try:
    measurement1 = np.loadtxt('data/Measurement1.asc')
    print("Successfully loaded 'data/Measurement1.asc'.")
except FileNotFoundError:
    print("Warning: 'data/Measurement1.asc' not found. Using dummy data for demonstration.")
    # Create dummy data: 200 samples, 3 channels
    np.random.seed(0)
    n_samples = 200
    n_channels_dummy = 3
    measurement1 = np.zeros((n_samples, n_channels_dummy))
    measurement1[0] = np.random.rand(n_channels_dummy) * 10

    # Create more realistic time series with some patterns and randomness
    for i in range(1, n_samples):
        measurement1[i] = 0.7 * measurement1[i - 1] + np.random.randn(n_channels_dummy) * 0.5 + np.sin(i / 10) * 0.3

    import os

    if not os.path.exists('data'):
        os.makedirs('data')
    np.savetxt('data/Measurement1.asc', measurement1)
    print(f"Dummy data with {n_samples} samples and {n_channels_dummy} channels created and saved.")
except Exception as e:
    print(f"Error loading data: {e}. Please check the file path and format.")
    exit()

# Create DataFrame
num_channels = measurement1.shape[1]
ch_names = [f'ch{i + 1}' for i in range(num_channels)]
df = pd.DataFrame(measurement1, columns=ch_names)
print(f"\nData loaded into DataFrame with {num_channels} channels and {len(df)} observations.")

# --- 1. Check Data ---
print("\n--- 1. Check Data ---")
print(df.head())
print(df.describe())

# --- 2. Check Stationarity ---
print("\n--- 2. Check Stationarity ---")
adf_kpss_results, all_stationary = check_stationarity(df) # Combined check
print(adf_kpss_results)

if not all_stationary:
    print("\nWarning: Data is not stationary according to KPSS tests. Differencing will be applied.")
    df_diff = df.diff().dropna()
    adf_kpss_results_after_diff, all_stationary_after_diff = check_stationarity(df_diff) # Combined check after diff
    print(adf_kpss_results_after_diff)
    if not all_stationary_after_diff:
        print("\nError: Data is still not stationary after differencing.  VAR may not be appropriate.")
        exit()
    else:
        df_model = df_diff
else:
    df_model = df

# --- 3. Detrend Data ---
print("\n--- 3. Detrend Data ---")
df_detrended = detrend_dataframe(df_model)
print("Detrended Data (first 5 rows):")
print(df_detrended.head())

# --- 4. Apply Moving Average (Filtering) ---
print("\n--- 4. Apply Moving Average (Filtering) ---")
window_size = 5
df_filtered = apply_moving_average(df_detrended, window=window_size)
print(f"Filtered Data (first 5 rows after removing NaN):")
print(df_filtered.head())

# --- 4b. Apply Smoothing (Optional) ---
smooth_window = 500
df_smoothed = smooth_dataframe(df_filtered, window_size=smooth_window)
print(f"\n--- 4b. Apply Smoothing (Window={smooth_window}) ---")
print(f"Smoothed Data (first 5 rows after removing NaN):\n{df_smoothed.head()}")

# --- 5. Scale Data ---
print("\n--- 5. Scale Data ---")
df_scaled = scale_dataframe(df_smoothed, method='standardize') # Use smoothed data for scaling
print("Scaled Data (first 5 rows):")
print(df_scaled.head())

# --- 6. Train-Test Split ---
print("\n--- 6. Train-Test Split ---")
split_ratio = 0.8
n_obs = len(df_scaled)
train_size = int(n_obs * split_ratio)
train_df = df_scaled.iloc[:train_size]
test_df = df_scaled.iloc[train_size:]
print(f"Train set: {len(train_df)} observations, Test set: {len(test_df)} observations")

# --- 7. VAR Model Order Selection ---
print("\n--- 7. VAR Model Order Selection ---")
max_lag_experiment = min(10, int(len(train_df) / 2))
min_lag = 1

if max_lag_experiment >= 1:
    model_selection = VAR(train_df)

    try:
        best_aic = float('inf')
        best_lag = 1

        print(f"\nTesting lag orders from {min_lag} to {max_lag_experiment}:")
        for lag in range(min_lag, max_lag_experiment + 1):
            try:
                results = model_selection.fit(lag)
                current_aic = results.aic
                print(f"  Lag {lag}: AIC = {current_aic:.4f}")

                if current_aic < best_aic:
                    best_aic = current_aic
                    best_lag = lag
            except Exception as e:
                print(f"  Error fitting lag {lag}: {e}")
                continue

        print(f"\nSelected lag order: {best_lag} (AIC: {best_aic:.4f})")
        final_selected_lag = best_lag
    except Exception as e:
        print(f"Error during lag selection: {e}")
        print("Defaulting to lag order 1")
        final_selected_lag = 1
else:
    print("Warning: Not enough data for lag selection. Defaulting to lag order 1.")
    final_selected_lag = 1

# --- 8. Fit VAR Model ---
print("\n--- 8. VAR Model Fitting ---")
model = VAR(train_df)
model_fitted = model.fit(final_selected_lag)
print(f"VAR({final_selected_lag}) model fitted successfully.")

# Optional: Print model diagnostics
print("\nGranger Causality Tests (shows relationships between variables):")
for c in df_scaled.columns:
    test_result = model_fitted.test_causality(c, df_scaled.columns.drop(c).tolist())
    print(f"{c} Granger-causes {test_result.causing}: p-value = {test_result.pvalue:.4f}")

# --- 9. Forecasting (Step-by-Step) ---
print("\n--- 9. Forecasting (Step-by-Step) ---")
lag_order = model_fitted.k_ar
history = train_df.values.tolist()
forecast_values_oos = []

for _ in range(len(test_df)):
    forecast_input = np.array(history[-lag_order:])
    forecast = model_fitted.forecast(y=forecast_input, steps=1)
    forecast_values_oos.append(forecast[0])
    history.append(forecast[0].tolist())

forecast_df_oos = pd.DataFrame(forecast_values_oos, index=test_df.index, columns=test_df.columns)

# --- 10. Inverse Transform the Forecasts ---
print("\n--- 10. Inverse Transform the Forecasts ---")

def inverse_scale_dataframe(df_scaled, original_df, method='standardize'):
    """
    Inverse scales the columns of a Pandas DataFrame back to the original scale.

    Args:
        df_scaled (pd.DataFrame): The scaled DataFrame.
        original_df (pd.DataFrame): The original DataFrame *before* scaling.
        method (str): The scaling method used. 'standardize' or 'minmax'.
            Default is 'standardize'.

    Returns:
        pd.DataFrame: The inverse-scaled DataFrame.
    """
    df_original_scale = df_scaled.copy()
    if method == 'standardize':
        for col in df_original_scale.columns:
            df_original_scale[col] = (df_original_scale[col].values * original_df[col].std()) + original_df[col].mean()
    elif method == 'minmax':
        for col in df_original_scale.columns:
            original_min = original_df[col].min()
            original_max = original_df[col].max()
            df_original_scale[col] = (df_original_scale[col].values * (original_max - original_min)) + original_min
    else:
        raise ValueError("Invalid scaling method. Choose 'standardize' or 'minmax'.")
    return df_original_scale

# Inverse scale using the statistics of the *smoothed* data
forecast_df_original_scale = inverse_scale_dataframe(forecast_df_oos, df_smoothed, method='standardize')

# Account for detrending and differencing
if not all_stationary:
    # Reverse differencing: Add the last value from the *original* data
    forecast_df_original_scale = forecast_df_original_scale.cumsum() + df.iloc[train_size + window_size - 1].values

# --- 11. Evaluate Forecasts (Proper Out-of-Sample) ---
print("\n--- 11. Evaluate Forecasts (Proper Out-of-Sample) ---")
evaluation_metrics_oos = pd.DataFrame(
    columns=['Channel', 'MSE', 'MAE', 'RMSE', 'R²']
)

# Align the test set with the forecast indices
# Account for rows lost due to smoothing and moving average
rows_lost = smooth_window - 1 + window_size - 1  # Rows lost due to smoothing and moving average
adjusted_train_size = train_size + rows_lost  # Adjust train_size to account for NaN rows dropped

# Ensure the test set matches the forecast length
test_df_aligned = df.iloc[adjusted_train_size:adjusted_train_size + len(forecast_df_original_scale)]

# Verify lengths match
if len(test_df_aligned) != len(forecast_df_original_scale):
    print(f"Error: Length mismatch - Actual test set: {len(test_df_aligned)}, Forecasted: {len(forecast_df_original_scale)}")
    exit()

for i, col in enumerate(test_df.columns):
    mse = mean_squared_error(test_df_aligned[col].values, forecast_df_original_scale[col].values)
    mae = mean_absolute_error(test_df_aligned[col].values, forecast_df_original_scale[col].values)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_df_aligned[col].values, forecast_df_original_scale[col].values)
    evaluation_metrics_oos.loc[i] = [col, mse, mae, rmse, r2]

print(evaluation_metrics_oos)

# --- 12. Visualization (Proper Out-of-Sample Forecasts) ---
print("\n--- 12.1 Visualization (Combined Plot) ---")
plt.figure(figsize=(15, 10))
for i, col in enumerate(df.columns):
    plt.subplot(len(df.columns), 1, i + 1)
    plt.plot(df.index[:train_size], df[col][:train_size], label='Training Data', color='blue')
    plt.plot(test_df_aligned.index, test_df_aligned[col], label='Actual Test Data', color='green')
    plt.plot(forecast_df_original_scale.index, forecast_df_original_scale[col], label='Forecast (Smoothed Train)', color='red', linestyle='--')
    plt.title(f"{col} - VAR({final_selected_lag}) Forecast (Smoothed Train)")
    plt.legend()
    plt.tight_layout()
plt.show()

# --- 12.2. Visualization (Proper Out-of-Sample Forecasts - Separate Plots) ---
print("\n--- 12.2 Visualization (Separate Plots) ---")
num_channels = df.shape[1]
fig, axes = plt.subplots(nrows=num_channels, ncols=1, figsize=(15, 5 * num_channels))

for i, col in enumerate(df.columns):
    ax = axes[i] if num_channels > 1 else axes
    ax.plot(test_df_aligned.index, test_df_aligned[col], label='Actual Test Data', color='green')
    ax.plot(forecast_df_original_scale.index, forecast_df_original_scale[col], label='Forecast (Smoothed Train)', color='red', linestyle='--')
    ax.set_title(f"{col} - Actual vs. Forecast (Smoothed Train) - VAR({final_selected_lag})")
    ax.set_xlabel('Time')
    ax.set_ylabel('Acceleration')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# --- 12.3 Visualization (Separate Figures with Transparency) ---
print("\n--- 12.3 Visualization (Separate Figures with Transparency) ---")
for i, col in enumerate(df.columns):
    plt.figure(figsize=(15, 6))
    plt.plot(test_df_aligned.index, test_df_aligned[col], label='Actual Test Data', color='green')
    plt.plot(forecast_df_original_scale.index, forecast_df_original_scale[col], label='Forecast (Smoothed Train)', color='red', linestyle='--', alpha=0.5)
    plt.title(f"{col} - Actual vs. Forecast (Smoothed Train) - VAR({final_selected_lag})")
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
