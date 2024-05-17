import pandas as pd

# Load the datasets from CSV files
data1 = pd.read_csv("./data/preprocessed_player_stats_Trial4.csv")
data2 = pd.read_csv("./data/preprocessed_player_stats.csv")

# Identify missing columns in data2 and create a DataFrame with these columns filled with zeros
missing_columns = list(set(data1.columns) - set(data2.columns))
missing_df = pd.DataFrame(0, index=data2.index, columns=missing_columns)

# Concatenate data2 with the missing_df to add the missing columns
data2 = pd.concat([data2, missing_df], axis=1)

# Ensure the order of columns in data2 matches data1
data2 = data2[data1.columns]

# Concatenate the datasets
combined_data = pd.concat([data1, data2])

# Save the combined dataset to a new CSV file
combined_data.to_csv("./data/combined_data.csv", index=False)

print("Datasets have been combined and saved to 'combined_data.csv'")
