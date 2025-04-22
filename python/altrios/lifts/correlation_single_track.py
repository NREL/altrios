# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import os
#
# file_name = "npf_performance_results.xlsx"
# if not os.path.exists(file_name):
#     raise FileNotFoundError("Please check file dictionary.")
#
# df = pd.read_excel(file_name)
#
# # columns_of_interest = [
# #     "daily_throughput(K)",
# #     "train_batch_size(k)",
# #     "M", "N", "n_t", "n_p", "n_r",
# #     "crane_numbers", "hostler_numbers",
# #     "ic_avg_processing_time", "oc_avg_processing_time", "total_avg_processing_time",
# #     "ic_avg_energy", "oc_avg_energy", "total_avg_energy"
# # ]
#
# columns_of_interest = [
#     "daily_throughput(K)",
#     "train_batch_size(k)",
#     "M", "N", "n_t", "n_p", "n_r",
#     "total_processing_time", "hostler_numbers",
#     "ic_avg_processing_time", "oc_avg_processing_time", "total_avg_processing_time",
#     "ic_avg_energy", "oc_avg_energy", "total_avg_energy"
# ]
#
# df_selected = df[columns_of_interest]
#
# correlation_matrix = df_selected.corr(method='pearson')
#
# plt.figure(figsize=(14, 12))
# sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
# plt.title("Correlation Heatmap of Selected Operational Parameters")
# plt.xticks(rotation=45, ha="right")
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.savefig("selected_parameters_correlation_heatmap.png", dpi=300)
# plt.show()


# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error
#
# file_name = "npf_performance_results.xlsx"
# df = pd.read_excel(file_name)
#
# # input: features
# features = ['daily_throughput(K)', 'train_batch_size(k)', 'M', 'N', 'n_t', 'n_p', 'n_r']
# X = df[features]
#
# # output: processing time & energy
# y = df['total_avg_processing_time']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
#
# # Model analysis
# r2 = r2_score(y_test, y_pred)
# rmse = mean_squared_error(y_test, y_pred)
#
# print("Model coefficient:")
# for feature, coef in zip(features, model.coef_):
#     print(f"{feature}: {coef:.4f}")
#
# print(f"\nIntercept: {model.intercept_:.4f}")
# print(f"R²: {r2:.4f}")
# print(f"RMSE: {rmse:.4f}")
#
# import matplotlib.pyplot as plt
#
# plt.scatter(df['train_batch_size(k)'], df['total_avg_processing_time'], label='True value')
# plt.plot(df['train_batch_size(k)'], model.predict(X), color='red', label='Fitting value')
# plt.xlabel('Train Batch Size (k)')
# plt.ylabel('Total Avg Processing Time')
# plt.title('Fitting plot of Train Batch Size vs. Total Avg Processing Time')
# plt.legend()
# plt.show()



