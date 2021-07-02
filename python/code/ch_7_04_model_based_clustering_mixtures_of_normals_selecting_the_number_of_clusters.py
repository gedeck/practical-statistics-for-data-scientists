#!/usr/local/bin/python

## Practical Statistics for Data Scientists (Python)
## Chapter 7. Unsupervised Learning
# > (c) 2019 Peter C. Bruce, Andrew Bruce, Peter Gedeck

import math
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.stats import multivariate_normal
import prince
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import from_levels_and_colors
import seaborn as sns
import common

print("""  ### Measures of Dissimilarity
""")
print("sp500_px = pd.read_csv('sp500_data.csv.gz', index_col=0)")
sp500_px = pd.read_csv(common.SP500_DATA_CSV, index_col=0)

df = sp500_px.loc[sp500_px.index >= '2011-01-01', ['XOM', 'CVX']]
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(5, 5))

# This causes TypeError: object of type 'float' has no len()
# for i, method in enumerate(['single', 'average', 'complete', 'ward']):
#     ax = axes[i // 2, i % 2]
#     Z = linkage(df, method=method)
#     colors = [f'C{c+1}' for c in fcluster(Z, 4, criterion='maxclust')]
#     ax = sns.scatterplot(x='XOM', y='CVX', hue=colors, style=colors,
#                          size=0.5, ax=ax, data=df, legend=False)
#
#     ax.set_xlim(-3, 3)
#     ax.set_ylim(-3, 3)
#     ax.set_title(method)
#
# plt.tight_layout()
# plt.show()

## Model based clustering
### Multivariate Normal Distribution
# > Define a colormap that corresponds to the probability levels

mean = [0.5, -0.5]
cov = [[1, 1], [1, 2]]
probability = [.5, .75, .95, .99]
def probLevel(p):
    D = 1
    return (1 - p) / (2 * math.pi * D)
levels = [probLevel(p) for p in probability]

fig, ax = plt.subplots(figsize=(5, 5))

x, y = np.mgrid[-2.8:3.8:.01, -5:4:.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv = multivariate_normal(mean, cov)

CS = ax.contourf(x, y, rv.pdf(pos), cmap=cm.GnBu, levels=50)
ax.contour(CS, levels=levels, colors=['black'])
ax.plot(*mean, color='black', marker='o')

plt.tight_layout()
plt.show()

print("  ### Mixtures of Normals")

df = sp500_px.loc[sp500_px.index >= '2011-01-01', ['XOM', 'CVX']]
mclust = GaussianMixture(n_components=2).fit(df)
print(mclust.bic(df))

fig, ax = plt.subplots(figsize=(4, 4))
colors = [f'C{c}' for c in mclust.predict(df)]
df.plot.scatter(x='XOM', y='CVX', c=colors, alpha=0.5, ax=ax)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

plt.tight_layout()
plt.show()

print('Mean')
print(mclust.means_)
print('Covariances')
print(mclust.covariances_)
print()

print("""  ### Selecting the number of clusters
""")
results = []
covariance_types = ['full', 'tied', 'diag', 'spherical']
for n_components in range(1, 9):
    for covariance_type in covariance_types:
        mclust = GaussianMixture(n_components = n_components, warm_start=True,
                                 covariance_type = covariance_type)
        mclust.fit(df)
        results.append({
            'bic': mclust.bic(df),
            'n_components': n_components,
            'covariance_type': covariance_type,
        })

results = pd.DataFrame(results)

colors = ['C0', 'C1', 'C2', 'C3']
styles = ['C0-','C1:','C0-.', 'C1--']

fig, ax = plt.subplots(figsize=(4, 4))
for i, covariance_type in enumerate(covariance_types):
    subset = results.loc[results.covariance_type == covariance_type, :]
    subset.plot(x='n_components', y='bic', ax=ax, label=covariance_type,
                kind='line', style=styles[i]) # , color=colors[i])

plt.tight_layout()
print("plt.show()")
plt.show()



# Next 300 scaling and categorical
