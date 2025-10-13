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

print( "  ### ROC Curve" )
print( "  # The function 'roc_curve' in Scikit-learn calculates all the information that is required for plotting a ROC curve." )
print( "  # (receiver operating characteristic curve)")
print()

fpr, tpr, thresholds = roc_curve(y, logit_reg.predict_proba(X)[:, 0],
                                 pos_label='default')
roc_df = pd.DataFrame({'recall': tpr, 'specificity': 1 - fpr})

ax = roc_df.plot(x='specificity', y='recall', figsize=(4, 4), legend=False)
ax.set_ylim(0, 1)
ax.set_xlim(1, 0)
ax.plot((1, 0), (0, 1))
ax.set_xlabel('specificity')
ax.set_ylabel('recall')
plt.tight_layout()
plt.show()

print( "  ### AUC" )
print( "  # Accuracy can easily be calculated using the scikit-learn function 'accuracy_score'." )
print()

print(np.sum(roc_df.recall[:-1] * np.diff(1 - roc_df.specificity)))
print(roc_auc_score([1 if yi == 'default' else 0 for yi in y], logit_reg.predict_proba(X)[:, 0]))

fpr, tpr, thresholds = roc_curve(y, logit_reg.predict_proba(X)[:,0],
                                 pos_label='default')
roc_df = pd.DataFrame({'recall': tpr, 'specificity': 1 - fpr})

ax = roc_df.plot(x='specificity', y='recall', figsize=(4, 4), legend=False)
ax.set_ylim(0, 1)
ax.set_xlim(1, 0)
# ax.plot((1, 0), (0, 1))
ax.set_xlabel('specificity')
ax.set_ylabel('recall')
ax.fill_between(roc_df.specificity, 0, roc_df.recall, alpha=0.3)
plt.tight_layout()
plt.show()
print()

print( "  ## Strategies for Imbalanced Data" )
print( "  ### Undersampling" )
print( "  # The results differ from the R version, however are equivalent to results obtained using the R code. ")
print( "  # Model based results are of similar magnitude." )
print()
print("full_train_set = pd.read_csv('full_train_set.csv.gz')")
full_train_set = pd.read_csv(common.FULL_TRAIN_SET_CSV)
print(full_train_set.shape)

print('percentage of loans in default: ',
print(      100 * np.mean(full_train_set.outcome == 'default')))

predictors = ['payment_inc_ratio', 'purpose_', 'home_', 'emp_len_',
              'dti', 'revol_bal', 'revol_util']
outcome = 'outcome'
X = pd.get_dummies(full_train_set[predictors], prefix='', prefix_sep='',
                   drop_first=True)
y = full_train_set[outcome]

full_model = LogisticRegression(penalty='l2', C=1e42, solver='liblinear')
full_model.fit(X, y)
print('percentage of loans predicted to default: ',
print(      100 * np.mean(full_model.predict(X) == 'default')))

(np.mean(full_train_set.outcome == 'default') /
 np.mean(full_model.predict(X) == 'default'))
print()

print( "  ### Oversampling and Up/Down Weighting" )
print()

default_wt = 1 / np.mean(full_train_set.outcome == 'default')
wt = [default_wt if outcome == 'default' else 1 for outcome in full_train_set.outcome]

full_model = LogisticRegression(penalty="l2", C=1e42, solver='liblinear')
full_model.fit(X, y, wt)
print('percentage of loans predicted to default (weighting): ',
print(      100 * np.mean(full_model.predict(X) == 'default')))
print()

print( "  ### Data Generation" )
print( "  # The package imbalanced-learn provides an implementation of the SMOTE and similar algorithms." )
print()

X_resampled, y_resampled = SMOTE().fit_resample(X, y)
print('percentage of loans in default (SMOTE resampled): ',
      100 * np.mean(y_resampled == 'default'))

full_model = LogisticRegression(penalty="l2", C=1e42, solver='liblinear')
full_model.fit(X_resampled, y_resampled)
print('percentage of loans predicted to default (SMOTE): ',
      100 * np.mean(full_model.predict(X) == 'default'))

X_resampled, y_resampled = ADASYN().fit_resample(X, y)
print('percentage of loans in default (ADASYN resampled): ',
      100 * np.mean(y_resampled == 'default'))

full_model = LogisticRegression(penalty="l2", C=1e42, solver='liblinear')
full_model.fit(X_resampled, y_resampled)
print('percentage of loans predicted to default (ADASYN): ',
print(      100 * np.mean(full_model.predict(X) == 'default')))


# Next 412 exploring predictions
