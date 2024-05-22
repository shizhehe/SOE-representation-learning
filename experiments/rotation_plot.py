# Let's combine all the steps into one block of code that generates the above plot.

# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Extracted data from the image provided by the user
# The data is structured as a dictionary with tuples as keys
# Tuples are structured as (Centering value, Number of Neighbors)
f1_data = {
    (0, 0): (0.794, 0.781), 
    (0, "15-45"): (0.600, 0.648),
    (0, "(90, 180, 270)"): (0.393, 0.297), 
    ("15-45", "15-45"): (0.679, 0.572), 
    ("15-45", "(90, 180, 270)"): (0.355, 0.371), 
    ("(90, 180, 270)", "(90, 180, 270)"): (0.684, 0.568), 
    ("(90, 180, 270)", "15-45"): (0.554, 0.213), 
    ("15-45", 0): (0.751, 0.638), 
    ("(90, 180, 270)", 0): (0.778, 0.622)
}

bacc_data = {
    (0, 0): (0.838, 0.823), 
    (0, "15-45"): (0.691, 0.728),
    (0, "(90, 180, 270)"): (0.552, 0.526), 
    ("15-45", "15-45"): (0.755, 0.716), 
    ("15-45", "(90, 180, 270)"): (0.575, 0.549), 
    ("(90, 180, 270)", "(90, 180, 270)"): (0.732, 0.671), 
    ("(90, 180, 270)", "15-45"): (0.511, 0.558), 
    ("15-45", 0): (0.801, 0.714), 
    ("(90, 180, 270)", 0): (0.821, 0.713)
}

data = f1_data

sns.set(font_scale=1.6)

# Convert the dictionary into a DataFrame
df = pd.DataFrame(list(data.items()), columns=['Index', 'Values'])
df[['Centering', 'Neighbors']] = pd.DataFrame(df['Index'].tolist(), index=df.index)
df[['BACC > c', 'BACC < c']] = pd.DataFrame(df['Values'].tolist(), index=df.index)
df.drop(columns=['Index', 'Values'], inplace=True)

# Calculate the absolute difference and percentage difference
df['Absolute Difference'] = df['BACC > c'] - df['BACC < c']
df['Percentage Difference'] = (df['Absolute Difference'] / df['BACC > c']) * 100

print(df)

# Pivot the DataFrame to create a matrix suitable for a heatmap of percentage differences
#percentage_heatmap_data = df.pivot("Centering", "Neighbors", "Percentage Difference")
percentage_heatmap_data = df.pivot_table(index="Centering", columns="Neighbors", values="Percentage Difference", aggfunc='mean')

# Create the heatmap using seaborn with a diverging color map
plt.figure(figsize=(12, 9))
ax = sns.heatmap(percentage_heatmap_data, annot=True, fmt=".2f",
                 cmap='PuBu', cbar_kws={'label': 'F1 % Absolute Difference SOE/No Pretraining'},
                 annot_kws={'size': 26}) #RdYlGn
plt.title('Robustness Testing - % F1 Gap SOE vs. No Pretraining', fontsize=28)
plt.ylabel('Model Trained on Rotations', fontsize=28)
plt.xlabel('Model Evaluated on Rotations', fontsize=28)

ax.tick_params(labelsize=30)

# Set the color bar font size
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=26)

modern_plot_path = 'robustness_experiment_f1.png'
plt.savefig(modern_plot_path, bbox_inches='tight', dpi=300)
# Display the heatmap
plt.show()

# This code block combines data extraction, transformation, percentage calculation, and visualization into one.