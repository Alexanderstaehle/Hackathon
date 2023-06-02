import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# read txt file into pandas
data = pd.read_csv('data/y_hat.txt', header=None)
# convert to array
data = data.to_numpy()
# Reshape the list into a 50x50 matrix
matrix = np.reshape(data, (50, 50))

# Create a heatmap of the matrix
plt.imshow(matrix, cmap='viridis')
plt.colorbar()
plt.show()