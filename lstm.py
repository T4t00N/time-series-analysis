
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- 1. Load Data ---
print("--- 1. Load Data ---")
measurement1 = np.loadtxt('data/Measurement1.asc')
num_channels = measurement1.shape[1]
ch_names = [f'ch{i + 1}' for i in range(num_channels)]
df = pd.DataFrame(measurement1, columns=ch_names)
print(f"Shape of the DataFrame: {df.shape}")
print(f"Number of channels: {num_channels}")
print(f"Channel names: {ch_names}")
print(df.head())

time_steps = 10  # Number of time steps for LSTM input

# --- 2. Data Preprocessing ---
print("\n--- 2. Data Preprocessing ---")
# Normalize the data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=ch_names)
print("Data normalized using MinMaxScaler.")

channel_name = f"channel_{i + 1}"  # Create a descriptive name for each channel
df_scaled[channel_name] = []   # Assign an empty list to the channel name


for i in range(len(df) - time_steps):
    for channel in range(num_channels):
        channel.append(df_scaled.iloc[i: (i + time_steps), channel])
    # Create the input sequence (X)
    X.append(df_scaled[i: (i + time_steps)])
    # Create the target value (y)
    y.append(df_scaled[i + time_steps])