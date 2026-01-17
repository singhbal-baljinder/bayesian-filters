import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define a path using 
path_to_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "examples", "vfe_results.csv")
# Load the data
df = pd.read_csv(path_to_data)

plt.figure(figsize=(10, 6))

# Plot Noisy Measurements
plt.scatter(df['time'], df['true_pos'], color='red', alpha=0.3, s=10, label='Measured (Noisy)')

# Plot KF Estimate
plt.plot(df['time'], df['est_pos'], color='blue', linewidth=2, label='KF Estimate (Position)')

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
