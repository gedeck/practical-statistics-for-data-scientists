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
# But this program hangs on plt.show() if you do that - graphical windows which you have to terminate manually
from dmba import plotDecisionTree, textDecisionTree
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import common

# Set this if the notebook crashes in the XGBoost part.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print()
print( "  ## Bagging and the Random Forest")
print()
print( "  ### Random Forest")
print()

print("loan3000 = pd.read_csv('loan3000.csv')")
loan3000 = pd.read_csv(common.LOAN3000_CSV)
print("loan_data = pd.read_csv('loan_data.csv.gz')")
loan_data = pd.read_csv(common.LOAN_DATA_CSV)
predictors = ['borrower_score', 'payment_inc_ratio']
outcome = 'outcome'

X = loan3000[predictors]
y = loan3000[outcome]

rf = RandomForestClassifier(n_estimators=500, random_state=1,
                            oob_score=True)
rf.fit(X, y)
print("print(rf.oob_decision_function_)")
print(rf.oob_decision_function_)

n_estimator = list(range(20, 510, 5))
oobScores = []
for n in n_estimator:
    rf = RandomForestClassifier(n_estimators=n,
                                criterion='entropy', max_depth=5,
                                random_state=1, oob_score=True)
    rf.fit(X, y)
    oobScores.append(rf.oob_score_)

pd.DataFrame({
    'n': n_estimator,
    'oobScore': oobScores
}).plot(x='n', y='oobScore')

predictions = X.copy()
predictions['prediction'] = rf.predict(X)
predictions.head()

fig, ax = common.printx("fig, ax = ", "plt.subplots(figsize=(4, 4))", {'plt':plt} )

predictions.loc[predictions.prediction=='paid off'].plot(
    x='borrower_score', y='payment_inc_ratio', style='.',
    markerfacecolor='none', markeredgecolor='C1', ax=ax)
predictions.loc[predictions.prediction=='default'].plot(
    x='borrower_score', y='payment_inc_ratio', style='o',
    markerfacecolor='none', markeredgecolor='C0', ax=ax)
ax.legend(['paid off', 'default']);
ax.set_xlim(0, 1)
ax.set_ylim(0, 25)
ax.set_xlabel('borrower_score')
ax.set_ylabel('payment_inc_ratio')

plt.tight_layout()
print("plt.show()")
plt.show()

print()
print("  ### Variable importance")
print("  # This is different to R. The accuracy decrease is not available out of the box.")
print()

predictors = ['loan_amnt', 'term', 'annual_inc', 'dti',
              'payment_inc_ratio', 'revol_bal', 'revol_util',
              'purpose', 'delinq_2yrs_zero', 'pub_rec_zero',
              'open_acc', 'grade', 'emp_length', 'purpose_',
              'home_', 'emp_len_', 'borrower_score']
outcome = 'outcome'

X = pd.get_dummies(loan_data[predictors], drop_first=True)
y = loan_data[outcome]

rf_all = RandomForestClassifier(n_estimators=500, random_state=1)
rf_all.fit(X, y)

rf_all_entropy = RandomForestClassifier(n_estimators=500, random_state=1,
                                        criterion='entropy')
print(rf_all_entropy.fit(X, y))

rf = RandomForestClassifier(n_estimators=500)
scores = defaultdict(list)

print("  # crossvalidate the scores on a number of different random splits of the data")
for _ in range(3):
    train_X, valid_X, train_y, valid_y = train_test_split(X, y,
                                                          test_size=0.3)
    rf.fit(train_X, train_y)
    acc = metrics.accuracy_score(valid_y, rf.predict(valid_X))
    for column in X.columns:
        X_t = valid_X.copy()
        X_t[column] = np.random.permutation(X_t[column].values)
        shuff_acc = metrics.accuracy_score(valid_y, rf.predict(X_t))
        scores[column].append((acc-shuff_acc)/acc)
print()
print('Features sorted by their score:')
print(sorted([(round(np.mean(score), 4), feat) for
              feat, score in scores.items()], reverse=True))

importances = rf_all.feature_importances_

df = pd.DataFrame({
    'feature': X.columns,
    'Accuracy decrease': [np.mean(scores[column]) for column in
                         X.columns],
    'Gini decrease': rf_all.feature_importances_,
    'Entropy decrease': rf_all_entropy.feature_importances_,
})
df = df.sort_values('Accuracy decrease')
fig, axes = plt.subplots(ncols=2, figsize=(8, 5))

fig, ax = common.printx("fig, ax = ", "plt.subplots(ncols=2, figsize=(8, 5))", {'plt':plt} )
ax = df.plot(kind='barh', x='feature', y='Accuracy decrease',
             legend=False, ax=axes[0])
ax.set_ylabel('')

ax = df.plot(kind='barh', x='feature', y='Gini decrease',
             legend=False, ax=axes[1])
ax.set_ylabel('')
ax.get_yaxis().set_visible(False)

plt.tight_layout()
print("plt.show()")
plt.show()
