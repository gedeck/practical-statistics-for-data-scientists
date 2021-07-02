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
from collections import Counter
import common

print("""  ## Scaling and Categorical Variables
  ### Scaling the Variables
""")
print("loan_data = pd.read_csv('loan_data.csv.gz')")
loan_data = pd.read_csv(common.LOAN_DATA_CSV)
print("sp500_px = pd.read_csv('sp500_data.csv.gz', index_col=0)")
sp500_px = pd.read_csv(common.SP500_DATA_CSV, index_col=0)
print()
loan_data['outcome'] = pd.Categorical(loan_data['outcome'],
                                      categories=['paid off', 'default'],
                                      ordered=True)
defaults = loan_data.loc[loan_data['outcome'] == 'default',]

columns = ['loan_amnt', 'annual_inc', 'revol_bal', 'open_acc',
           'dti', 'revol_util']

df = defaults[columns]
kmeans = KMeans(n_clusters=4, random_state=1).fit(df)
counts = Counter(kmeans.labels_)

centers = pd.DataFrame(kmeans.cluster_centers_, columns=columns)
centers['size'] = [counts[i] for i in range(4)]
print("print(centers)")
print(centers)

scaler = preprocessing.StandardScaler()
df0 = scaler.fit_transform(df * 1.0)

kmeans = KMeans(n_clusters=4, random_state=1).fit(df0)
counts = Counter(kmeans.labels_)

centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),
                       columns=columns)
centers['size'] = [counts[i] for i in range(4)]
print("print(centers)")
print(centers)
print()

print("  ### Dominant Variables")
syms = ['GOOGL', 'AMZN', 'AAPL', 'MSFT', 'CSCO', 'INTC', 'CVX', 'XOM',
        'SLB', 'COP', 'JPM', 'WFC', 'USB', 'AXP', 'WMT', 'TGT', 'HD', 'COST']
top_sp1 = sp500_px.loc[sp500_px.index >= '2005-01-01', syms]

sp_pca1 = PCA()
sp_pca1.fit(top_sp1)
explained_variance = pd.DataFrame(sp_pca1.explained_variance_)
ax = explained_variance.head(10).plot.bar(legend=False, figsize=(4, 4))
ax.set_xlabel('Component')

plt.tight_layout()
print("plt.show()")
plt.show()
print()
print("""loadings = pd.DataFrame(sp_pca1.components_[0:2, :],
                        columns=top_sp1.columns)
print(loadings.transpose())""")
loadings = pd.DataFrame(sp_pca1.components_[0:2, :],
                        columns=top_sp1.columns)
print(loadings.transpose())

print("""  ### Categorical Data and Gower's Distance
  # Currently not available in any of the standard packages.
  # However work is in progress to add it to scikit-learn. We will update this notebook once it becomes available
  #
  # https://github.com/scikit-learn/scikit-learn/pull/9555/
""")
x = defaults[['dti', 'payment_inc_ratio', 'home_', 'purpose_']].loc[0:4, :]
print(x)

# ```
#
################################################################
### Figure 7-13: Categorical data and Gower's distance
#
# x <- loan_data[1:5, c('dti', 'payment_inc_ratio', 'home_', 'purpose_')]
# x
#
# daisy(x, metric='gower')
#
# set.seed(301)
# df <- loan_data[sample(nrow(loan_data), 250),
#                 c('dti', 'payment_inc_ratio', 'home_', 'purpose_')]
# d = daisy(df, metric='gower')
# hcl <- hclust(d)
# dnd <- as.dendrogram(hcl)
#
# png(filename=file.path(PSDS_PATH, 'figures', 'psds_0713.png'), width = 4, height=4, units='in', res=300)
# par(mar=c(0,5,0,0)+.1)
# plot(dnd, leaflab='none', ylab='distance')
# dev.off()
#
# dnd_cut <- cut(dnd, h=.5)
# df[labels(dnd_cut$lower[[1]]),]
#
#
### Problems in clustering with mixed data types
# df <- model.matrix(~ -1 + dti + payment_inc_ratio + home_ + pub_rec_zero, data=defaults)
# df0 <- scale(df)
# km0 <- kmeans(df0, centers=4, nstart=10)
# centers0 <- scale(km0$centers, center=FALSE, scale=1/attr(df0, 'scaled:scale'))
# round(scale(centers0, center=-attr(df0, 'scaled:center'), scale=FALSE), 2)
# ```

print("""  ### Problems with Clustering Mixed Data
""")
print("""columns = ['dti', 'payment_inc_ratio', 'home_', 'pub_rec_zero']
df = pd.get_dummies(defaults[columns])

scaler = preprocessing.StandardScaler()

df0 = scaler.fit_transform(df * 1.0)
kmeans = KMeans(n_clusters=4, random_state=1).fit(df0)
centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),
                       columns=df.columns)
print(centers)""")
columns = ['dti', 'payment_inc_ratio', 'home_', 'pub_rec_zero']
df = pd.get_dummies(defaults[columns])

scaler = preprocessing.StandardScaler()

df0 = scaler.fit_transform(df * 1.0)
kmeans = KMeans(n_clusters=4, random_state=1).fit(df0)
centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),
                       columns=df.columns)
print(centers)
