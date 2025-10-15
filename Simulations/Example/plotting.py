import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from typing import Dict, Any


def process_nested_data(results: Dict[str, Any], category: str) -> pd.DataFrame:
    sigmas = results.get('path_loss_variances', [])
    payload_sizes = results.get('payload_sizes', [])
    data_dict = results.get(category, {})

    all_data = []

    for s in sigmas:
        for p in payload_sizes:
            try:
                data_point = data_dict[str(p)][str(s)]

                if isinstance(data_point, pd.DataFrame):
                    row_dict = data_point.iloc[0].to_dict()
                elif isinstance(data_point, pd.Series):
                    row_dict = data_point.to_dict()
                else:
                    continue

                row_dict['Sigma'] = s
                row_dict['Payload_Size'] = p
                all_data.append(row_dict)
            except KeyError:
                print(f"  Peringatan: Data tidak ditemukan untuk {category}, payload={p}, sigma={s}")
                continue

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    cols_to_front = ['Sigma', 'Payload_Size']
    df = df[cols_to_front + [col for col in df.columns if col not in cols_to_front]]
    return df


def process_energy_data(results: Dict[str, Any]) -> pd.DataFrame:
    print("Memproses data untuk kategori: 'energy'...")
    sigmas = results.get('path_loss_variances', [])
    payload_sizes = results.get('payload_sizes', [])

    all_energy_data = []
    for s in sigmas:
        for p in payload_sizes:
            try:
                mean_e = results['mean_energy'][str(p)][str(s)]
                std_e = results['std_energy'][str(p)][str(s)]
                all_energy_data.append({
                    'Sigma': s,
                    'Payload_Size': p,
                    'Mean_Energy': mean_e,
                    'Std_Energy': std_e
                })
            except KeyError:
                continue

    return pd.DataFrame(all_energy_data)


sns.set(style="whitegrid", context="talk")
color_palette = sns.color_palette("tab20", 20)
file = './results/True_True_0.01_cnst_num_bytes_itm.p'
output_dir = "./results/csv_output/"
prop_model = "ITM"

os.makedirs(output_dir, exist_ok=True)

with open(file, "rb") as f:
    results = pickle.load(f)

categories_to_process = ['nodes', 'gateway', 'air_interface']
dataframes = {}

for category in categories_to_process:
    df_category = process_nested_data(results, category)
    if not df_category.empty:
        dataframes[category] = df_category
        csv_path = os.path.join(output_dir, f"{category}_results.csv")
        df_category.to_csv(csv_path, index=False)

df_energy = process_energy_data(results)
if not df_energy.empty:
    dataframes['energy'] = df_energy
    csv_path = os.path.join(output_dir, "energy_summary.csv")
    df_energy.to_csv(csv_path, index=False)

df_nodes = dataframes.get('nodes')

plot_metric = "TotalEnergy"
plt.figure(figsize=(12, 8))

sigmas = df_nodes['Sigma'].unique()
payload_sizes = sorted(df_nodes['Payload_Size'].unique())

for idx, s in enumerate(sigmas):
    subset_df = df_nodes[df_nodes['Sigma'] == s]
    subset_df = subset_df.sort_values(by='Payload_Size')

    y_vals = subset_df[plot_metric]
    x_vals = subset_df['Payload_Size']

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

plot_path = os.path.join(output_dir, f'plot_{plot_metric}_{prop_model}.png')
plt.savefig(plot_path)
plt.show()
