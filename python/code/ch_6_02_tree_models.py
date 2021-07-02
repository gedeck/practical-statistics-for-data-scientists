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
from dmba import plotDecisionTree, textDecisionTree
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import common

# Set this if the notebook crashes in the XGBoost part.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("  ## Tree Models")
print("  ### A Simple Example")
print("  # The package scikit-learn has the class DecisionTreeClassifier to build a decision tree model. ")
print("  # The function plotDecisionTree from the dmba package can be used to visualize the tree.")
print()

print("loan3000 = pd.read_csv('loan3000.csv')")
loan3000 = pd.read_csv(common.LOAN3000_CSV)
print("""predictors = ['borrower_score', 'payment_inc_ratio']
outcome = 'outcome'
X = loan3000[predictors]
y = loan3000[outcome]
loan_tree = DecisionTreeClassifier(random_state=1, criterion='entropy',
                                   min_impurity_decrease=0.003)
loan_tree.fit(X, y)
plotDecisionTree(loan_tree, feature_names=predictors, class_names=loan_tree.classes_)
print(textDecisionTree(loan_tree))""")
predictors = ['borrower_score', 'payment_inc_ratio']
outcome = 'outcome'
X = loan3000[predictors]
y = loan3000[outcome]
loan_tree = DecisionTreeClassifier(random_state=1, criterion='entropy',
                                   min_impurity_decrease=0.003)
loan_tree.fit(X, y)
plotDecisionTree(loan_tree, feature_names=predictors, class_names=loan_tree.classes_)
print(textDecisionTree(loan_tree))
