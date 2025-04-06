import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

file_name = "npf_performance_results.xlsx"
if not os.path.exists(file_name):
    raise FileNotFoundError("Please check file dictionary.")

df = pd.read_excel(file_name)

# todo: correlation between time / energy & others
columns_of_interest = [
    "daily_throughput(K)",
    "train_batch_size(k)",
    "M", "N", "n_t", "n_p", "n_r",
    "crane_numbers", "hostler_numbers",
    "ic_avg_processing_time", "oc_avg_processing_time", "total_avg_processing_time",
    "ic_avg_energy", "oc_avg_energy", "total_avg_energy"
]

df_selected = df[columns_of_interest]

correlation_matrix = df_selected.corr(method='pearson')

plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
plt.title("Correlation Heatmap of Selected Operational Parameters")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("selected_parameters_correlation_heatmap.png", dpi=300)
plt.show()
