import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define a path using 
path_to_data_ekf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "examples", "ekf_results.csv")
path_to_data_vfe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "examples", "vfe_results.csv")
# Load the data
df_ekf = pd.read_csv(path_to_data_ekf)
df_vfe = pd.read_csv(path_to_data_vfe)

plt.figure(figsize=(10, 6))

# Plot Noisy Measurements
plt.scatter(df_ekf['time'], df_ekf['true_pos'], color='red', alpha=0.3, s=10, label='Measured (Noisy)')

# Plot KF Estimate
plt.plot(df_vfe['time'], df_vfe['true_pos'], color='blue', linewidth=3, label='True value (Position)')
plt.plot(df_ekf['time'], df_ekf['est_pos'], color='red', linewidth=2, label='EKF Estimate (Position)')
plt.plot(df_vfe['time'], df_vfe['est_pos'], color='black', linewidth=2, linestyle='--', label='VFE Estimate (Position)')

# Plot Uncertainty Bounds (3-Sigma)
#plt.fill_between(df['time'], 
#                 df['est_pos'] - 3*df['std_pos'], 
#                 df['est_pos'] + 3*df['std_pos'], 
#                 color='blue', alpha=0.2, label='3$\sigma$ Confidence')

plt.title('Kalman Filter Tracking: Position')
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.legend()
plt.grid(True)
plt.show()
