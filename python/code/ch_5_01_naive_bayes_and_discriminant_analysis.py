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
# $ pip install imblearn
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from pygam import LinearGAM, s, f, l
# If you uncomment this line, it has the side-effect of making matplotlib.pyplot ptl.show() fail with "no display found. Using non-interactive Agg backend"
# from dmba import classificationSummary
import seaborn as sns
import matplotlib.pyplot as plt
import common

print("""  ## Naive Bayes
  ### The Naive Solution
""")
print("loan_data = pd.read_csv('loan_data.csv')")
loan_data = pd.read_csv(common.LOAN_DATA_CSV)
print()

print("  # convert to categorical ")
print("""loan_data.outcome = loan_data.outcome.astype('category')
loan_data.outcome.cat.reorder_categories(['paid off', 'default'])
loan_data.purpose_ = loan_data.purpose_.astype('category')
loan_data.home_ = loan_data.home_.astype('category')
loan_data.emp_len_ = loan_data.emp_len_.astype('category')
predictors = ['purpose_', 'home_', 'emp_len_']
outcome = 'outcome'
X = pd.get_dummies(loan_data[predictors], prefix='', prefix_sep='')
y = loan_data[outcome]""")
loan_data.outcome = loan_data.outcome.astype('category')
loan_data.outcome.cat.reorder_categories(['paid off', 'default'])
loan_data.purpose_ = loan_data.purpose_.astype('category')
loan_data.home_ = loan_data.home_.astype('category')
loan_data.emp_len_ = loan_data.emp_len_.astype('category')
predictors = ['purpose_', 'home_', 'emp_len_']
outcome = 'outcome'
X = pd.get_dummies(loan_data[predictors], prefix='', prefix_sep='')
y = loan_data[outcome]

print("""naive_model = MultinomialNB(alpha=0.01, fit_prior=True)
naive_model = MultinomialNB(alpha=0, fit_prior=False)
naive_model.fit(X, y)
new_loan = X.loc[146:146, :]
print('predicted class: ', naive_model.predict(new_loan)[0])""")
naive_model = MultinomialNB(alpha=0.01, fit_prior=True)
naive_model = MultinomialNB(alpha=0, fit_prior=False)
naive_model.fit(X, y)
new_loan = X.loc[146:146, :]
print('predicted class: ', naive_model.predict(new_loan)[0])

probabilities = common.printx("probabilities = ",
                              "pd.DataFrame(naive_model.predict_proba(new_loan),columns=naive_model.classes_)",
                              {'pd':pd, 'naive_model': naive_model, 'new_loan':new_loan} )
print('predicted probabilities ',)
print(probabilities)

#### Example not in book

# Numerical variables are not supported in scikit-learn. The example would need to demonstrate binning a variable and display the probability distribution of the bins.
# ```
### example not in book
# less_naive <- NaiveBayes(outcome ~ borrower_score + payment_inc_ratio +
#                            purpose_ + home_ + emp_len_, data = loan_data)
# less_naive$table[1:2]
#
# png(filename=file.path(PSDS_PATH, 'figures', 'psds_naive_bayes.png'),  width = 4, height=3, units='in', res=300)
#
# stats <- less_naive$table[[1]]
# ggplot(data.frame(borrower_score=c(0,1)), aes(borrower_score)) +
#   stat_function(fun = dnorm, color='blue', linetype=1,
#                 arg=list(mean=stats[1, 1], sd=stats[1, 2])) +
#   stat_function(fun = dnorm, color='red', linetype=2,
#                 arg=list(mean=stats[2, 1], sd=stats[2, 2])) +
#   labs(y='probability')
# dev.off()
# ```

print("""
  ## Discriminant Analysis
  ### A Simple Example
""")

print("loan3000 = pd.read_csv('loan3000.csv')")
loan3000 = pd.read_csv(common.LOAN3000_CSV)
print("""loan3000.outcome = loan3000.outcome.astype('category')
predictors = ['borrower_score', 'payment_inc_ratio']
outcome = 'outcome'
X = loan3000[predictors]
y = loan3000[outcome]
loan_lda = LinearDiscriminantAnalysis()
loan_lda.fit(X, y)
print(pd.DataFrame(loan_lda.scalings_, index=X.columns))

pred = pd.DataFrame(loan_lda.predict_proba(loan3000[predictors]),
                    columns=loan_lda.classes_)
print(pred.head())""")
loan3000.outcome = loan3000.outcome.astype('category')
predictors = ['borrower_score', 'payment_inc_ratio']
outcome = 'outcome'
X = loan3000[predictors]
y = loan3000[outcome]
loan_lda = LinearDiscriminantAnalysis()
loan_lda.fit(X, y)
print(pd.DataFrame(loan_lda.scalings_, index=X.columns))

pred = pd.DataFrame(loan_lda.predict_proba(loan3000[predictors]),
                    columns=loan_lda.classes_)
print(pred.head())

print("""
  #### Figure 5.1
  # Use scalings and center of means to determine decision boundary
""")
center = np.mean(loan_lda.means_, axis=0)
slope = - loan_lda.scalings_[0] / loan_lda.scalings_[1]
intercept = center[1] - center[0] * slope

print("  # payment_inc_ratio for borrower_score of 0 and 20")
print("""x_0 = (0 - intercept) / slope
x_20 = (20 - intercept) / slope
lda_df = pd.concat([loan3000, pred['default']], axis=1)
lda_df.head()""")
x_0 = (0 - intercept) / slope
x_20 = (20 - intercept) / slope
lda_df = pd.concat([loan3000, pred['default']], axis=1)
lda_df.head()

fig, ax = common.printx("fig, ax = ", "plt.subplots(figsize=(4, 4))", {'plt':plt} )
g = common.printx("g = ", """sns.scatterplot(x='borrower_score', y='payment_inc_ratio',
                    hue='default', data=lda_df,
                    palette=sns.diverging_palette(240, 10, n=9, as_cmap=True),
                    ax=ax, legend=False)""",
                    {'sns':sns, 'ax':ax, 'lda_df':lda_df} )
print("""ax.set_ylim(0, 20)
ax.set_xlim(0.15, 0.8)
ax.plot((x_0, x_20), (0, 20), linewidth=3)
ax.plot(*loan_lda.means_.transpose())
plt.tight_layout()
plt.show()""")
ax.set_ylim(0, 20)
ax.set_xlim(0.15, 0.8)
ax.plot((x_0, x_20), (0, 20), linewidth=3)
ax.plot(*loan_lda.means_.transpose())
plt.tight_layout()
plt.show()
