import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from Simulations.GlobalConfig import prop_model

sns.set(style="whitegrid", context="talk")

color_palette = sns.color_palette("tab20", 20)

file = './results/True_True_0.01_cnst_num_bytes_itm.p'

results = pickle.load(open(file, "rb"))

path_loss_variances = results['path_loss_variances']
sigmas = results['path_loss_variances']
payload_sizes = results['payload_sizes']
nodes = results['nodes']

all_node_data = []

for s in sigmas:
    for p in payload_sizes:
        node_df = nodes[str(p)][str(s)]
        row_dict = node_df.iloc[0].to_dict()

        row_dict['Sigma'] = s
        row_dict['Payload_Size'] = p

        all_node_data.append(row_dict)

df = pd.DataFrame(all_node_data)

cols_to_front = ['Sigma', 'Payload_Size']
df = df[cols_to_front + [col for col in df.columns if col not in cols_to_front]]

output_dir = "./results/"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "simulation_results.csv")
df.to_csv(csv_path, index=False)

print(df)
print("\n" + "=" * 50 + "\n")

plot_metric = "TotalEnergy"
plt.figure(figsize=(12, 8))

for idx, s in enumerate(sigmas):
    x_vals = []
    y_vals = []

    for p in payload_sizes:
        filtered_data = df[(df['Sigma'] == s) & (df['Payload_Size'] == p)]

        retransmitted_bytes = filtered_data['RetransmittedPackets'].iloc[0] * p
        collided_bytes = filtered_data['CollidedPackets'].iloc[0] * p
        unique_bytes = filtered_data['UniquePackets'].iloc[0] * p

        if plot_metric == "TotalBytes":
            y = filtered_data['TotalBytes'].iloc[0]
        elif plot_metric == "RetransmittedPackets":
            y = retransmitted_bytes
        elif plot_metric == "CollidedPackets":
            y = collided_bytes
        elif plot_metric == "UniquePackets":
            y = unique_bytes
        elif plot_metric == "TotalEnergy":
            y = filtered_data['TotalEnergy'].iloc[0]
        else:
            continue

        x_vals.append(p)
        y_vals.append(y)

    plt.plot(x_vals, y_vals,
             marker='o',
             linestyle='-',
             color=color_palette[idx % len(color_palette)],
             label=f"σ = {s}")

plt.title(f"{plot_metric} vs Payload Size ({prop_model})", fontsize=16)
plt.xlabel("Payload Size (bytes)")
plt.ylabel(plot_metric.replace('_', ' '))
plt.legend(title="Path Loss Variance (σ)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

plot_path = f'./results/plot_{plot_metric}_{prop_model}.png'
plt.savefig(plot_path)
plt.show()