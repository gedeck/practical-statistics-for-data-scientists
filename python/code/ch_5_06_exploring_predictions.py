#!/usr/local/bin/python

## Practical Statistics for Data Scientists (Python)
## Chapter 5. Classification
# > (c) 2019 Peter C. Bruce, Andrew Bruce, Peter Gedeck

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression #, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from pygam import LinearGAM, s, f, l
# from dmba import classificationSummary
import seaborn as sns
import matplotlib.pyplot as plt
import common

print("loan_data = pd.read_csv('loan_data.csv')")
loan_data = pd.read_csv(common.LOAN_DATA_CSV)
print("""predictors = ['payment_inc_ratio', 'purpose_', 'home_', 'emp_len_',
              'borrower_score']""")
predictors = ['payment_inc_ratio', 'purpose_', 'home_', 'emp_len_',
              'borrower_score']
print("outcome = 'outcome'")
outcome = 'outcome'
X = common.printx("X = ", """pd.get_dummies(loan_data[predictors], prefix='', prefix_sep='',
                   drop_first=True)""", {'pd':pd, 'loan_data':loan_data, 'predictors':predictors})
y = common.printx( "y = ", "loan_data[outcome]", {'loan_data':loan_data, 'outcome':outcome} ) # .cat.categories
logit_reg = LogisticRegression(penalty='l2', C=1e42, solver='liblinear')
logit_reg.fit(X, y)
print()

print("  ### Exploring the Predictions")
print()

print("loan3000 = pd.read_csv('loan3000.csv')")
loan3000 = pd.read_csv(common.LOAN3000_CSV)
print("""predictors = ['borrower_score', 'payment_inc_ratio']
outcome = 'outcome'
X = loan3000[predictors]
y = loan3000[outcome]""")
predictors = ['borrower_score', 'payment_inc_ratio']
outcome = 'outcome'
X = loan3000[predictors]
y = loan3000[outcome]
print("""loan_tree = DecisionTreeClassifier(random_state=1, criterion='entropy',
                                   min_impurity_decrease=0.003)
loan_tree.fit(X, y)""")
loan_tree = DecisionTreeClassifier(random_state=1, criterion='entropy',
                                   min_impurity_decrease=0.003)
loan_tree.fit(X, y)

loan_lda = LinearDiscriminantAnalysis()
loan_lda.fit(X, y)
logit_reg = LogisticRegression(penalty="l2", solver='liblinear')
logit_reg.fit(X, y)
print()
print("  ## model")
print("""
gam = LinearGAM(s(0) + s(1))
print(gam.gridsearch(X.values, [1 if yi == 'default' else 0 for yi in y]))""")
gam = LinearGAM(s(0) + s(1))
print(gam.gridsearch(X.values, [1 if yi == 'default' else 0 for yi in y]))
print("""models = {
    'Decision Tree': loan_tree,
    'Linear Discriminant Analysis': loan_lda,
    'Logistic Regression': logit_reg,
    'Generalized Additive Model': gam,
}""")
models = {
    'Decision Tree': loan_tree,
    'Linear Discriminant Analysis': loan_lda,
    'Logistic Regression': logit_reg,
    'Generalized Additive Model': gam,
}

fig, axes = common.printx("fig, axes = ", "plt.subplots(nrows=2, ncols=2, figsize=(5, 5))", {'plt':plt} )

xvalues = np.arange(0.25, 0.73, 0.005)
yvalues = np.arange(-0.1, 20.1, 0.1)
xx, yy = np.meshgrid(xvalues, yvalues)
X = np.c_[xx.ravel(), yy.ravel()]

print("""
boundary = {}
for n, (title, model) in enumerate(models.items()):
    ...

plt.tight_layout()
plt.show()""")
boundary = {}
for n, (title, model) in enumerate(models.items()):
    ax = axes[n // 2, n % 2]
    predict = model.predict(X)
    if 'Generalized' in title:
        Z = np.array([1 if z > 0.5 else 0 for z in predict])
    else:

        Z = np.array([1 if z == 'default' else 0 for z in predict])
    Z = Z.reshape(xx.shape)
    boundary[title] = yvalues[np.argmax(Z > 0, axis=0)]
    boundary[title][Z[-1,:] == 0] = yvalues[-1]

    c = ax.pcolormesh(xx, yy, Z, cmap='Blues', vmin=0.1, vmax=1.3, shading='auto')
    ax.set_title(title)
    ax.grid(True)

plt.tight_layout()
plt.show()

boundary['borrower_score'] = xvalues
boundaries = pd.DataFrame(boundary)
fig, axes = common.printx("fig, axes = ", "plt.subplots(figsize=(5, 4))", {'plt':plt} )
print("""boundaries.plot(x='borrower_score', ax=ax)
ax.set_ylabel('payment_inc_ratio')
ax.set_ylim(0, 20)
plt.tight_layout()
plt.show()""")
boundaries.plot(x='borrower_score', ax=ax)
ax.set_ylabel('payment_inc_ratio')
ax.set_ylim(0, 20)
plt.tight_layout()
plt.show()
