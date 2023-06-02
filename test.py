import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

data = pd.read_csv('data/data.csv')
# standard scale the data
data = (data - data.mean()) / data.std()
# perform PCA on the data

pca = PCA(n_components=2, random_state=42)
pca.fit(data)
data = pca.transform(data)
# get the standard deviations of the two resulting dimensions
stds = np.std(data, axis=0)
print(stds)
# plot the data and flip it upside down
plt.scatter(data[:, 0], data[:, 1])
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.show()
