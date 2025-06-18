import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data for the Performance table
data_performance = {
    "Factor": ["Original Status", "Q-learning", "Deep Q Network", "MARL-Centralized", "MARL-Decentralized"],
    "Duration": [266.77, 47.64, 46.78, 43.89, 41.32],
    "Real-Time Factor": [167.012, 172.935, 183.492, 194.128, 207.645],
    "UPS": [117244, 119586, 123438, 127902, 132154],
    "UPS - Person": [684, 7360, 7895, 8412, 8965]
}

# Creating a DataFrame
df_performance = pd.DataFrame(data_performance)

# Setting up the positions for the bars
x = np.arange(len(df_performance['Factor']))  # the label locations
width = 0.2  # the width of the bars

# Color mapping for each model
colors = {
    'Original Status': 'red',
    'Q-learning': 'blue',
    'Deep Q Network': 'gold',
    'MARL-Centralized': 'green',
    'MARL-Decentralized': 'brown'
}

# Creating a figure with 2 rows and 2 columns
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Adding black borders around the graphs
for ax in axs.flat:
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

# Plot 1: Duration
for i, factor in enumerate(df_performance['Factor']):
    axs[0, 0].bar(x[i], df_performance['Duration'][i], width, color=colors[factor])
axs[0, 0].set_title('Duration')
axs[0, 0].set_ylabel('Duration (in seconds)')
axs[0, 0].set_xticks(x)
axs[0, 0].set_xticklabels(df_performance['Factor'], rotation=45, ha='right', fontsize=10)

# Plot 2: Real-Time Factor
for i, factor in enumerate(df_performance['Factor']):
    axs[0, 1].bar(x[i], df_performance['Real-Time Factor'][i], width, color=colors[factor])
axs[0, 1].set_title('Real-Time Factor')
axs[0, 1].set_ylabel('Real-Time Factor')
axs[0, 1].set_xticks(x)
axs[0, 1].set_xticklabels(df_performance['Factor'], rotation=45, ha='right', fontsize=10)

# Plot 3: UPS
for i, factor in enumerate(df_performance['Factor']):
    axs[1, 0].bar(x[i], df_performance['UPS'][i], width, color=colors[factor])
axs[1, 0].set_title('UPS')
axs[1, 0].set_ylabel('UPS')
axs[1, 0].set_xticks(x)
axs[1, 0].set_xticklabels(df_performance['Factor'], rotation=45, ha='right', fontsize=10)

# Plot 4: UPS - Person
for i, factor in enumerate(df_performance['Factor']):
    axs[1, 1].bar(x[i], df_performance['UPS - Person'][i], width, color=colors[factor])
axs[1, 1].set_title('UPS - Person')
axs[1, 1].set_ylabel('UPS - Person')
axs[1, 1].set_xticks(x)
axs[1, 1].set_xticklabels(df_performance['Factor'], rotation=45, ha='right', fontsize=10)

# Automatically adjust the layout to avoid overlap
plt.tight_layout()

# Display the plots
plt.show()

# Displaying the table
print("Performance Table:")
print(df_performance)
