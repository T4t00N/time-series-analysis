import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")  # Suppress routine warnings

# --- 0. Load Data (using your existing approach) ---
try:
    measurement1 = np.loadtxt('data/Measurement2.asc')
    print("Successfully loaded 'data/Measurement1.asc'.")
except FileNotFoundError:
    print("Warning: 'data/Measurement1.asc' not found. Using dummy data for demonstration.")
    # Create dummy data: 200 samples, 3 channels
    np.random.seed(0)  # for reproducibility
    n_samples = 200
    n_channels_dummy = 3
    measurement1 = np.zeros((n_samples, n_channels_dummy))
    measurement1[0] = np.random.rand(n_channels_dummy) * 10

    # Create more realistic time series with some patterns and randomness
    for i in range(1, n_samples):
        # AR(1) process with noise and slight trend
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


# --- 2. Train-Test Split ---
print("\n--- 2. Train-Test Split ---")
split_ratio = 0.8
n_obs = len(df)
train_size = int(n_obs * split_ratio)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]
print(f"Train set: {len(train_df)} observations, Test set: {len(test_df)} observations")

# --- 3. VAR Model Order Selection ---
print("\n--- 3. VAR Model Order Selection ---")
max_lag_experiment = min(400, int(len(train_df) / 2)) # Increased max_lag
min_lag = 400

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
    print("Warning: Not enough data for lag selection. Defaulting to lag 1.")
    final_selected_lag = 1

# --- 4. Fit VAR Model ---
print("\n--- 4. VAR Model Fitting ---")
model = VAR(train_df)
model_fitted = model.fit(final_selected_lag)
print(f"VAR({final_selected_lag}) model fitted successfully.")

# Optional: Print model diagnostics
print("\nGranger Causality Tests (shows relationships between variables):")
for c in df.columns:
    test_result = model_fitted.test_causality(c, df.columns.drop(c).tolist())
    print(f"{c} Granger-causes {test_result.causing}: p-value = {test_result.pvalue:.4f}")

# --- 5. Forecasting (Step-by-Step) ---
print("\n--- 5. Forecasting ---")
lag_order = model_fitted.k_ar
history = train_df.values.tolist()
forecast_values_oos = []

for _ in range(len(test_df)):
    forecast_input = np.array(history[-lag_order:])
    forecast = model_fitted.forecast(y=forecast_input, steps=1)
    forecast_values_oos.append(forecast[0])
    history.append(forecast[0].tolist()) # Append the *forecasted* value

forecast_df_oos = pd.DataFrame(forecast_values_oos, index=test_df.index, columns=test_df.columns)



# --- 6. Evaluate Forecasts (Proper Out-of-Sample) ---
print("\n--- 6. Evaluate Forecasts (Proper Out-of-Sample) ---")
evaluation_metrics_oos = pd.DataFrame(
    columns=['Channel', 'MSE', 'MAE', 'RMSE', 'R²']
)

for i, col in enumerate(test_df.columns):
    mse = mean_squared_error(test_df[col], forecast_df_oos[col])
    mae = mean_absolute_error(test_df[col], forecast_df_oos[col])
    rmse = np.sqrt(mse)
    r2 = r2_score(test_df[col], forecast_df_oos[col])
    evaluation_metrics_oos.loc[i] = [col, mse, mae, rmse, r2]

print(evaluation_metrics_oos)

# --- 7. Visualization (Proper Out-of-Sample Forecasts) ---
print("\n--- 7.1 Visualization (Combined Plot) ---")
plt.figure(figsize=(15, 10))
for i, col in enumerate(df.columns):
    plt.subplot(len(df.columns), 1, i + 1)
    plt.plot(df.index[:train_size], df[col][:train_size], label='Training Data', color='blue')
    plt.plot(df.index[train_size:], df[col][train_size:], label='Actual Test Data', color='green')
    plt.plot(forecast_df_oos.index, forecast_df_oos[col], label='Forecast', color='blue', linestyle='--')
    plt.title(f'{col} - VAR({final_selected_lag}) Forecast (Out-of-Sample)')
    plt.legend()
plt.tight_layout()
plt.show()

# --- 7.2. Visualization (Proper Out-of-Sample Forecasts - Separate Plots) ---
print("\n--- 7.2 Visualization (Separate Plots) ---")
num_channels = df.shape[1]
fig, axes = plt.subplots(nrows=num_channels, ncols=1, figsize=(15, 5 * num_channels))

for i, col in enumerate(df.columns):
    ax = axes[i]
    ax.plot(test_df.index, test_df[col], label='Actual Test Data', color='green')
    ax.plot(forecast_df_oos.index, forecast_df_oos[col], label='Forecast', color='blue', linestyle='--')
    ax.set_title(f'{col} - Actual vs. Out-of-Sample Forecast (VAR({final_selected_lag}))')
    ax.set_xlabel('Time')
    ax.set_ylabel('Acceleration')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# --- 7.3 Visualization (Proper Out-of-Sample Forecasts - Separate Figures with Transparency) ---
print("\n--- 7.3 Visualization (Propereparate Figures with Transparency) ---")

for i, col in enumerate(df.columns):
    plt.figure(figsize=(15, 6))
    plt.plot(test_df.index, test_df[col], label='Actual Test Data', color='green')
    plt.plot(forecast_df_oos.index, forecast_df_oos[col], label='Forecast', color='blue', linestyle='--', alpha=0.5)
    plt.title(f'{col} - Actual vs. Out-of-Sample Forecast (VAR({final_selected_lag}))')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()