# notebooks/fft_visualization.py

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../fft_features_with_anomalies.csv")
df['window_start'] = pd.to_datetime(df['window_start'], errors='coerce')

# Example plot: total_power over time with anomaly highlights
plt.figure(figsize=(12,6))
plt.plot(df['window_start'], df['total_power'], label='Total Power')
plt.scatter(df[df['anomaly']==-1]['window_start'],
            df[df['anomaly']==-1]['total_power'],
            color='red', label='Anomalies')
plt.xlabel('Time')
plt.ylabel('Total Power')
plt.title('FFT Total Power with Anomalies')
plt.legend()
plt.show()
