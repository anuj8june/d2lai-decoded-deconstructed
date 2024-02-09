import torch
import pandas as pd

# Reading data file
data_fpath = "data/house_tiny.csv"
data = pd.read_csv(data_fpath)
print(f"\nCSV data output: \n{data}")

# Segregating dependent and indeoendent variable
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(f"\nInputs: \n{inputs}")

# Filling nan values with mean
inputs = inputs.fillna(inputs.mean())
print(f"\nInputs after nan fill: \n{inputs}")

# Converting to tensor format
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
print(f"X: \n{X}, \ny: \n{y}")

# Exersises are not that much useful for me for this particular section.