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

# Initial inspection
print("\nData info:")
print(df.info())
print("\nDescriptive statistics:")
# Change display settings to show all columns
pd.set_option('display.max_columns', None)

# Now describe() will print all columns
print(df.describe())


# Time Series Plots
plt.figure(figsize=(15, 10))
for i in range(10):  # 0 to 9
    plt.subplot(5, 2, i+1)
    plt.plot(df.iloc[:, i])
    plt.title(f'Accelerometer {i+1}')
    plt.xlabel('Time/Index')
    plt.ylabel('Acceleration')
    plt.ylim(-2, 2)  # Set y-axis limits to -2 and 2
plt.tight_layout()
plt.show()

for i in range(10):
    plt.figure(figsize=(7, 4))  # Adjust the figure size as desired
    plt.plot(df.iloc[:, i])  # Column 0 corresponds to Accelerometer 1
    plt.title(f'Channel {i+1}')
    plt.xlabel('Observation index')
    plt.ylabel('Acceleration')
    plt.ylim(-2, 2)  # Y-axis limits
    plt.grid(True)  # Optional: add grid for readabilit
    plt.savefig(f'channel_{i+1}.png')  # Save each plot as a PNG file
    plt.show()

# Histograms
plt.figure(figsize=(15, 10))
for i in range(10):
    plt.subplot(5, 2, i+1)
    sns.histplot(df.iloc[:, i], kde=True)
    plt.title(f'Accelerometer {i+1} Distribution')
plt.tight_layout()
#plt.show()

# Correlation Matrix
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Accelerometers')
#plt.show()

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Example of imputation (linear interpolation) - only run if you have missing data
# data = data.interpolate()