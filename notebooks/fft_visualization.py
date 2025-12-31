# notebooks/fft_visualization.ipynb

import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("../fft_features_with_anomalies.csv")

# Convert timestamps to datetime
df['window_start'] = pd.to_datetime(df['window_start'], errors='coerce')
df['window_end'] = pd.to_datetime(df['window_end'], errors='coerce')

# List of instruments
instruments = df['Instrument Code'].unique()

# Use a modern colormap
colors = plt.colormaps['tab20'].resampled(len(instruments))

# Plot total_power over time for each instrument
plt.figure(figsize=(14, 6))

for i, inst in enumerate(instruments):
    inst_df = df[df['Instrument Code'] == inst]
    plt.plot(inst_df['window_start'], inst_df['total_power'],
             label=inst, color=colors(i))

    # Highlight anomalies
    anomalies = inst_df[inst_df['anomaly'] == -1]
    plt.scatter(anomalies['window_start'], anomalies['total_power'],
                color=colors(i), edgecolor='red', marker='x', s=50, label=f"{inst} anomalies")

plt.xlabel('Time')
plt.ylabel('Total Power')
plt.title('FFT Total Power with Anomalies')
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()
