#!/usr/local/bin/python

## Practical Statistics for Data Scientists (Python)
## Chapter 6. Statistical Machine Learning
# > (c) 2019 Peter C. Bruce, Andrew Bruce, Peter Gedeck

import math
import os
import random
from pathlib import Path
from collections import defaultdict
from itertools import product
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# You have to comment out the following line to enable matplotlib.pyplot to work
# from dmba import plotDecisionTree, textDecisionTree
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import common

# Set this if the notebook crashes in the XGBoost part.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print()
print( "  ## K-Nearest Neighbors" )
print( "  ### A Small Example: Predicting Loan Default" )
print()

print("loan200 = pd.read_csv('loan200.csv')")
loan200 = pd.read_csv(common.LOAN200_CSV)

predictors = ['payment_inc_ratio', 'dti']
outcome = 'outcome'

newloan = loan200.loc[0:0, predictors]
X = loan200.loc[1:, predictors]
y = loan200.loc[1:, outcome]

print("""knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X, y)
knn.predict(newloan)
prob = knn.predict_proba( newloan )
one = prob[0][0]
two = prob[0][1]
print("newloan is ", newloan)
print("Probability of payment of new loan ", one )
print("Probability of default of new loan ", two )""")
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X, y)
knn.predict(newloan)
prob = knn.predict_proba( newloan )
one = prob[0][0]
two = prob[0][1]
print("newloan is ", newloan)
print("Probability of payment of new loan ", one )
print("Probability of default of new loan ", two )
if( one > two ):
  print("The new loan will probably be paid")
else:
  print("The new loan will probably not be paid")

nbrs = knn.kneighbors(newloan) # ♬ Neighbors... everybody needs good neighbors... ♪♪♪
maxDistance = np.max(nbrs[0][0])
fig, ax = plt.subplots(figsize=(4, 4))
common.printx("", """sns.scatterplot(x='payment_inc_ratio', y='dti', style='outcome',
                hue='outcome', data=loan200, alpha=0.3, ax=ax)""",
                {'sns':sns, 'loan200':loan200, 'pd':pd, 'ax':ax, 'nbrs':nbrs} )
common.printx("", """sns.scatterplot(x='payment_inc_ratio', y='dti', style='outcome',
                hue='outcome',
                data=pd.concat([loan200.loc[0:0, :], loan200.loc[nbrs[1][0] + 1,:]]),
                ax=ax, legend=False)""",
                {'sns':sns, 'loan200':loan200, 'pd':pd, 'ax':ax, 'nbrs':nbrs} )
ellipse = common.printx("ellipse = ", """Ellipse(xy=newloan.values[0],
                  width=2 * maxDistance, height=2 * maxDistance,
                  edgecolor='black', fc='None', lw=1)""",
                  {'newloan':newloan, 'Ellipse':Ellipse, 'maxDistance':maxDistance} )
ax.add_patch(ellipse)
ax.set_xlim(3, 16)
ax.set_ylim(15, 30)
plt.tight_layout()
print("plt.show()")
plt.show()
print()

print( "  ### Standardization (Normalization, Z-Scores)" )
print()
print("loan_data = pd.read_csv('loan_data.csv.gz')")
loan_data = pd.read_csv(common.LOAN_DATA_CSV)
loan_data = loan_data.drop(columns=['Unnamed: 0', 'status'])
loan_data['outcome'] = pd.Categorical(loan_data['outcome'],
                                      categories=['paid off', 'default'],
                                      ordered=True)
predictors = ['payment_inc_ratio', 'dti', 'revol_bal', 'revol_util']
outcome = 'outcome'
newloan = loan_data.loc[0:0, predictors]
print("newloan ", newloan)
X = loan_data.loc[1:, predictors]
y = loan_data.loc[1:, outcome]

print("""knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
nbrs = knn.kneighbors(newloan)
print(X.iloc[nbrs[1][0], :])""")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
nbrs = knn.kneighbors(newloan)
print(X.iloc[nbrs[1][0], :])

newloan = loan_data.loc[0:0, predictors]
X = loan_data.loc[1:, predictors]
y = loan_data.loc[1:, outcome]

scaler = preprocessing.StandardScaler()
scaler.fit(X * 1.0)
X_std = scaler.transform(X * 1.0)
newloan_std = scaler.transform(newloan * 1.0)
print("""
  # The same, but with standardized values

scaler = preprocessing.StandardScaler()
scaler.fit(X * 1.0)
X_std = scaler.transform(X * 1.0)
newloan_std = scaler.transform(newloan * 1.0)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_std, y)
nbrs = knn.kneighbors(newloan_std)
print(X.iloc[nbrs[1][0], :])""")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_std, y)
nbrs = knn.kneighbors(newloan_std)
print(X.iloc[nbrs[1][0], :])
print()

print( "  ### KNN as a Feature Engine" )
print()
print("loan_data = pd.read_csv('loan_data.csv.gz')")
loan_data = pd.read_csv(common.LOAN_DATA_CSV)
loan_data = loan_data.drop(columns=['Unnamed: 0', 'status'])
loan_data['outcome'] = pd.Categorical(loan_data['outcome'],
                                      categories=['paid off', 'default'],
                                      ordered=True)
predictors = ['dti', 'revol_bal', 'revol_util', 'open_acc',
              'delinq_2yrs_zero', 'pub_rec_zero']
outcome = 'outcome'

X = loan_data[predictors]
y = loan_data[outcome]

print("""knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X, y)
plt.scatter(range(len(X)), [bs + random.gauss(0, 0.015) for bs in knn.predict_proba(X)[:,0]],
            alpha=0.1, marker='.')
knn.predict_proba(X)[:, 0]""")
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X, y)
plt.scatter(range(len(X)), [bs + random.gauss(0, 0.015) for bs in knn.predict_proba(X)[:,0]],
            alpha=0.1, marker='.')
knn.predict_proba(X)[:, 0]
plt.tight_layout()
print("plt.show()")
plt.show()

print("""
  # The likelihood a borrower will default based on his credit history

loan_data['borrower_score'] = knn.predict_proba(X)[:, 0]
print(loan_data['borrower_score'].describe())""")
loan_data['borrower_score'] = knn.predict_proba(X)[:, 0]
print(loan_data['borrower_score'].describe())
