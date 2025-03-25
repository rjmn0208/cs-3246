import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Sample dataset: each row is an observation, each column is a feature.
data = np.array([[5, 20],
                 [15, 30],
                 [10, 25]])

# Initialize MinMaxScaler to scale features to the range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit the scaler to the data and transform the data
normalized_data = scaler.fit_transform(data)

print("Original data:")
print(data)
print("\nNormalized data:")
print(normalized_data)
