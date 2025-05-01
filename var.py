import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
print("All imports imported")

# 1. Load data
measurement1 = np.loadtxt('Measurement1.asc')
ch_names = [f'ch{i+1}' for i in range(10)]
df = pd.DataFrame(measurement1, columns=ch_names)
print("Data loaded to Dataframe")
# 2. Check stationarity
is_stationary = True
for name, series in df.items():
    result = adfuller(series.dropna())
    p_value = result[1]
    print(f'p-value for {name}: {p_value:.4f}')
    if p_value > 0.05:
        print(f'⚠️ {name} might be non-stationary (p > 0.05)')
        is_stationary = False
    else:
        print(f'{name} appears stationary.')
    print('-' * 10)

# 3. Handle non-stationary data
if not is_stationary:
    print("\n⚠️ WARNING: Some series are non-stationary. Differencing data.")
    data_for_model = df.diff().dropna()
else:
    print("\n✅ All series appear stationary. Proceeding with original data.")
    data_for_model = df

# 4. Manually set lag order
chosen_lag_order = 1
print(f"\n--- Using Fixed Lag Order: {chosen_lag_order} ---")

# 5. Fit the VAR model
model = VAR(data_for_model)

try:
    var_results = model.fit(chosen_lag_order)
    print("\n--- VAR Model Fit Results ---")
    print(var_results.summary())
except Exception as e:
    print(f"\n❌ Error fitting VAR model: {e}")
    print("Check data size vs lag order or whether differencing created too few rows.")

print("\n✅ VAR model fitting complete.")
