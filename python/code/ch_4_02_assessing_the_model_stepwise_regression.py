#!/usr/local/bin/python

## Practical Statistics for Data Scientists (Python)
## Chapter 4. Regression and Prediction
# > (c) 2019 Peter C. Bruce, Andrew Bruce, Peter Gedeck

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence
# $ pip install pygam
from pygam import LinearGAM, s, l
from pygam.datasets import wage
import seaborn as sns
import matplotlib.pyplot as plt
from dmba import stepwise_selection
from dmba import AIC_score
import common

print( "  ### Assessing the Model" )
print( "  # Scikit-learn provides a number of metrics to determine the quality of a model. Here we use the 'r2_score'." )
print("")
print("house = pd.read_csv('house_sales.csv', sep='\\t')")
house = pd.read_csv(common.HOUSE_CSV, sep='\t')
predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms',
              'Bedrooms', 'BldgGrade']
outcome = 'AdjSalePrice'
house_lm = LinearRegression()
house_lm.fit(house[predictors], house[outcome])
fitted = house_lm.predict(house[predictors])
RMSE = np.sqrt(mean_squared_error(house[outcome], fitted))
r2 = r2_score(house[outcome], fitted)
print(f'RMSE: {RMSE:.0f}')
print(f'r2: {r2:.4f}')

print( """  # While scikit-learn provides a variety of different metrics,
  # statsmodels provides a more in-depth analysis of the linear regression model.
  # This package has two different ways of specifying the model, one that is similar to
  # scikit-learn and one that allows specifying R-style formulas. Here we use the first approach.
  # As statsmodels doesn't add an intercept automaticaly, we need to add a constant column with value
  # 1 to the predictors. We can use the pandas method assign for this.
""")
model = sm.OLS(house[outcome], house[predictors].assign(const=1))
results = model.fit()
print("print(results.summary())")
print(results.summary())
print()

print( "  ### Model Selection and Stepwise Regression" )
print()
predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms',
              'BldgGrade', 'PropertyType', 'NbrLivingUnits',
              'SqFtFinBasement', 'YrBuilt', 'YrRenovated',
              'NewConstruction']

X = pd.get_dummies(house[predictors], drop_first=True)
X['NewConstruction'] = [1 if nc else 0 for nc in X['NewConstruction']]

house_full = sm.OLS(house[outcome], X.assign(const=1))
results = house_full.fit()
print("print(results.summary())")
print(results.summary())

print( "  # We can use the 'stepwise_selection method from the _dmba_ package." )
print("")

y = house[outcome]

def train_model(variables):
    if len(variables) == 0:
        return None
    model = LinearRegression()
    model.fit(X[variables], y)
    return model

def score_model(model, variables):
    if len(variables) == 0:
        return AIC_score(y, [y.mean()] * len(y), model, df=1)
    return AIC_score(y, model.predict(X[variables]), model)

best_model, best_variables = stepwise_selection(X.columns, train_model, score_model,
                                                verbose=True)
print()
print(f'Intercept: {best_model.intercept_:.3f}')
print('Coefficients:')
for name, coef in zip(best_variables, best_model.coef_):
    print(f' {name}: {coef}')
