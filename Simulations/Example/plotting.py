import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from Simulations.GlobalConfig import *

# Visual style
sns.set(style="whitegrid", context="talk")

# Palette colors
color_palette = sns.color_palette("tab20", 20)

# Load simulation results
file = './results/True_True_0.01_cnst_num_bytes_itm.p'
results = pickle.load(open(file, "rb"))

path_loss_variances = results['path_loss_variances']
sigmas = results['path_loss_variances']
payload_sizes = results['payload_sizes']
nodes = results['nodes']

# Select the metrics you want to visualize
plot_metric = "TotalEnergy"  # change to: 'RetransmittedPackets', 'CollidedPackets', 'TotalEnergy', etc

# Figure initialization
plt.figure(figsize=(10, 6))

for idx, s in enumerate(sigmas):
    x_vals = []
    y_vals = []

    for p in payload_sizes:
        node_data = nodes[str(p)][str(s)]
        if plot_metric == "TotalBytes":
            y = node_data['TotalBytes']
        elif plot_metric == "RetransmittedPackets":
            y = node_data['RetransmittedPackets'] * p
        elif plot_metric == "CollidedPackets":
            y = node_data['CollidedPackets'] * p
        elif plot_metric == "UniquePackets":
            y = node_data['UniquePackets'] * p
        elif plot_metric == "TotalEnergy":
            y = node_data['TotalEnergy']
        else:
            continue

        x_vals.append(p)
        y_vals.append(y)

    plt.plot(x_vals, y_vals,
             marker='o',
             color=color_palette[idx % len(color_palette)],
             label=f"σ = {s}")

# Labels, titles, and legends
plt.title(f"Comparison of {plot_metric} against Payload Size ({prop_model})", fontsize=16)
plt.xlabel("Payload Size (bytes)")
plt.ylabel(plot_metric.replace('_', ' '))
# plt.legend(title="Path Loss Variance (σ)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.show()
df = pd.DataFrame(results['nodes'])
print(df)
output_dir = f"./results/std_energy_{prop_model}/"
df.to_csv(f"{output_dir}std_energy.csv", index=False)
plt.xlabel("Payload size")
plt.savefig(f'./results/plot_0.01_{prop_model}.png')
plt.show()