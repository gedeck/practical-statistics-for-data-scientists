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

loan_data.outcome = loan_data.outcome.astype('category')
loan_data.outcome.cat.reorder_categories(['paid off', 'default'])
loan_data.purpose_ = loan_data.purpose_.astype('category')
loan_data.home_ = loan_data.home_.astype('category')
loan_data.emp_len_ = loan_data.emp_len_.astype('category')

print("""  ## Logistic regression
  ### Logistic Response Function and Logit""")

p = np.arange(0.01, 1, 0.01)
df = pd.DataFrame({
    'p': p,
    'logit': np.log(p / (1 - p)),
    'odds': p / (1 - p),
})

fig, ax = common.printx("fig, ax = ", "plt.subplots(figsize=(3, 3))", {'plt':plt})
ax.axhline(0, color='grey', linestyle='--')
ax.axvline(0.5, color='grey', linestyle='--')
ax.plot(df['p'], df['logit'])
ax.set_xlabel('Probability')
ax.set_ylabel('logit(p)')
plt.tight_layout()
print("plt.show()")
plt.show()

print("""
  ###  Logistic Regression and the GLM
  # The package scikit-learn has a specialised class for 'LogisticRegression'.
  Statsmodels has a more general method based on generalized linear model (GLM).
""")
print("""predictors = ['payment_inc_ratio', 'purpose_', 'home_', 'emp_len_',
              'borrower_score']""")
predictors = ['payment_inc_ratio', 'purpose_', 'home_', 'emp_len_',
              'borrower_score']
print("outcome = 'outcome'")
outcome = 'outcome'
X = common.printx("X = ", """pd.get_dummies(loan_data[predictors], prefix='', prefix_sep='',
                   drop_first=True)""", {'pd':pd, 'loan_data':loan_data, 'predictors':predictors})

y = common.printx( "y = ", "loan_data[outcome]", {'loan_data':loan_data, 'outcome':outcome} )# .cat.categories
print("""logit_reg = LogisticRegression(penalty='l2', C=1e42, solver='liblinear')
logit_reg.fit(X, y)
logit_reg = LogisticRegression(penalty='l2', C=1e42, solver='liblinear')
logit_reg.fit(X, y)
print('intercept ', logit_reg.intercept_[0])""")
logit_reg = LogisticRegression(penalty='l2', C=1e42, solver='liblinear')
logit_reg.fit(X, y)
print('intercept ', logit_reg.intercept_[0])

print("""print('classes', logit_reg.classes_)""")
print('classes', logit_reg.classes_)
print()

pd.DataFrame({'coeff': logit_reg.coef_[0]},
             index=X.columns)

print("  # Note that the intercept and coefficients are reversed compared to the R model ")
print("""
print(loan_data['purpose_'].cat.categories)
print(loan_data['home_'].cat.categories)
print(loan_data['emp_len_'].cat.categories)""")
print(loan_data['purpose_'].cat.categories)
print(loan_data['home_'].cat.categories)
print(loan_data['emp_len_'].cat.categories)

print("""
  # Not in book :
  # If you have a feature or outcome variable that is ordinal, use the
  # scikit-learn OrdinalEncoder to replace the categories
  # (here, 'paid off' and 'default') with numbers. In the below code,
  # we replace 'paid off' with 0 and 'default' with 1.
  # This reverses the order of the predicted classes and as a consequence,
  # the coefficients will be reversed.""")
print("""
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder(categories=[['paid off', 'default']])
y_enc = enc.fit_transform(loan_data[[outcome]]).ravel()
logit_reg_enc = LogisticRegression(penalty="l2", C=1e42, solver='liblinear')
logit_reg_enc.fit(X, y_enc)""")
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder(categories=[['paid off', 'default']])
y_enc = enc.fit_transform(loan_data[[outcome]]).ravel()
logit_reg_enc = LogisticRegression(penalty="l2", C=1e42, solver='liblinear')
logit_reg_enc.fit(X, y_enc)
print("""print('intercept ', logit_reg_enc.intercept_[0])""")
print('intercept ', logit_reg_enc.intercept_[0])

print("""print('classes', logit_reg_enc.classes_)""")
print('classes', logit_reg_enc.classes_)
print()

pd.DataFrame({'coeff': logit_reg_enc.coef_[0]},
             index=X.columns)

print("  ### Predicted Values from Logistic Regression")
print("""
pred = pd.DataFrame(logit_reg.predict_proba(X),
                    columns=logit_reg.classes_)
print(pred.describe())""")
pred = pd.DataFrame(logit_reg.predict_proba(X),
                    columns=logit_reg.classes_)
print(pred.describe())
print()

print("  ### Interpreting the Coefficients and Odds Ratios")
print("""
fig, ax = plt.subplots(figsize=(3, 3))
ax.plot(df['logit'], df['odds'])
ax.set_xlabel('log(odds ratio)')
ax.set_ylabel('odds ratio')
ax.set_xlim(0, 5.1)
ax.set_ylim(-5, 105)
plt.tight_layout()
plt.show()""")
fig, ax = plt.subplots(figsize=(3, 3))
ax.plot(df['logit'], df['odds'])
ax.set_xlabel('log(odds ratio)')
ax.set_ylabel('odds ratio')
ax.set_xlim(0, 5.1)
ax.set_ylim(-5, 105)
plt.tight_layout()
plt.show()

# Next 220 - assessing the model
