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
from dmba import classificationSummary
import seaborn as sns
import matplotlib.pyplot as plt
import common

print()
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
y = common.printx( "y = ", "loan_data[outcome]", {'loan_data':loan_data, 'outcome':outcome} ) #
print()

print( "  ## Evaluating Classification Models" )
print( "  ### Confusion Matrix" )
print()

logit_reg = LogisticRegression(penalty='l2', C=1e42, solver='liblinear')
logit_reg.fit(X, y)
pred = logit_reg.predict(X)
pred_y = logit_reg.predict(X) == 'default'
true_y = y == 'default'
true_pos = true_y & pred_y
true_neg = ~true_y & ~pred_y
false_pos = ~true_y & pred_y
false_neg = true_y & ~pred_y

conf_mat = pd.DataFrame([[np.sum(true_pos), np.sum(false_neg)], [np.sum(false_pos), np.sum(true_neg)]],
                       index=['Y = default', 'Y = paid off'],
                       columns=['Yhat = default', 'Yhat = paid off'])
print(conf_mat)
print(confusion_matrix(y, logit_reg.predict(X)))
print()

print( "  # The package 'dmba' contains the function 'classificationSummary'" )
print( "  # that prints confusion matrix and accuracy for a classification model." )
print()
print("""classificationSummary(y, logit_reg.predict(X),
                      class_names=logit_reg.classes_)""")
classificationSummary(y, logit_reg.predict(X),
                      class_names=logit_reg.classes_)
print()

print( "  ### Precision, Recall, and Specificity" )
print( "  # The 'scikit-learn' function 'precision_recall_fscore_support' returns" )
print( "  # precision, recall, fbeta_score and support." )
print()

print("conf_mat = confusion_matrix(y, logit_reg.predict(X))")
conf_mat = confusion_matrix(y, logit_reg.predict(X))
print('Precision', conf_mat[0, 0] / sum(conf_mat[:, 0]))
print('Recall', conf_mat[0, 0] / sum(conf_mat[0, :]))
print('Specificity', conf_mat[1, 1] / sum(conf_mat[1, :]))
print("""print(precision_recall_fscore_support(y, logit_reg.predict(X),
                                labels=['default', 'paid off']) )""" )
print(precision_recall_fscore_support(y, logit_reg.predict(X),
                                labels=['default', 'paid off']) )
