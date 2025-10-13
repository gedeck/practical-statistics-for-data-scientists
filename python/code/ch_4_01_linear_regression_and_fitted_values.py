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
# from dmba import stepwise_selection
# from dmba import AIC_score
import common

print( "  ## Simple Linear Regression" )
print( "  ### The Regression Equation" )
print( "  ### Lung Disease" )
print()
print("lung = pd.read_csv('LungDisease.csv')")
lung = pd.read_csv(common.LUNG_CSV)
print("""lung.plot.scatter(x='Exposure', y='PEFR')
plt.show()
""")
lung.plot.scatter(x='Exposure', y='PEFR')
plt.tight_layout()
plt.show()

print( "  # We can use the 'LinearRegression' model from scikit-learn" )
print("""
predictors = ['Exposure']
outcome = 'PEFR'
model = LinearRegression()
model.fit(lung[predictors], lung[outcome])
print(f'Intercept: {model.intercept_:.3f}')
print(f'Coefficient Exposure: {model.coef_[0]:.3f}')""")
predictors = ['Exposure']
outcome = 'PEFR'
model = LinearRegression()
model.fit(lung[predictors], lung[outcome])
print(f'Intercept: {model.intercept_:.3f}')
print(f'Coefficient Exposure: {model.coef_[0]:.3f}')

fig, ax = common.printx("fig, ax = ", "plt.subplots(figsize=(4, 4))", {'plt': plt} )
print("""ax.set_xlim(0, 23)
ax.set_ylim(295, 450)
ax.set_xlabel('Exposure')
ax.set_ylabel('PEFR')
ax.plot((0, 23), model.predict([[0], [23]]))
ax.text(0.4, model.intercept_, r'$b_0$', size='larger')
x = [[7.5], [17.5]]
y = model.predict(x)
ax.plot((7.5, 7.5, 17.5), (y[0], y[1], y[1]), '--')
ax.text(5, np.mean(y), r'$\Delta Y$', size='larger')
ax.text(12, y[1] - 10, r'$\Delta X$', size='larger')
ax.text(12, 390, r'$b_1 = \\frac{\Delta Y}{\Delta X}$', size='larger')
plt.show()""")
ax.set_xlim(0, 23)
ax.set_ylim(295, 450)
ax.set_xlabel('Exposure')
ax.set_ylabel('PEFR')
ax.plot((0, 23), model.predict([[0], [23]]))
ax.text(0.4, model.intercept_, r'$b_0$', size='larger')
x = [[7.5], [17.5]]
y = model.predict(x)
ax.plot((7.5, 7.5, 17.5), (y[0], y[1], y[1]), '--')
ax.text(5, np.mean(y), r'$\Delta Y$', size='larger')
ax.text(12, y[1] - 10, r'$\Delta X$', size='larger')
ax.text(12, 390, r'$b_1 = \frac{\Delta Y}{\Delta X}$', size='larger')
plt.tight_layout()
plt.show()
print()

print( "  ### Fitted Values and Residuals" )
print( "  # The method 'predict' of a fitted scikit-learn model can be used to predict new data points." )
print()

print("""fitted = model.predict(lung[predictors])
residuals = lung[outcome] - fitted
ax = lung.plot.scatter(x='Exposure', y='PEFR', figsize=(4, 4))
ax.plot(lung.Exposure, fitted)
for x, yactual, yfitted in zip(lung.Exposure, lung.PEFR, fitted):
    ax.plot((x, x), (yactual, yfitted), '--', color='C1')
plt.show()""")
fitted = model.predict(lung[predictors])
residuals = lung[outcome] - fitted
ax = lung.plot.scatter(x='Exposure', y='PEFR', figsize=(4, 4))
ax.plot(lung.Exposure, fitted)
for x, yactual, yfitted in zip(lung.Exposure, lung.PEFR, fitted):
    ax.plot((x, x), (yactual, yfitted), '--', color='C1')
plt.tight_layout()
plt.show()
print()

print( "  ## Multiple linear regression" )
print("""  ### House sales

subset = ['AdjSalePrice', 'SqFtTotLiving', 'SqFtLot', 'Bathrooms',
          'Bedrooms', 'BldgGrade']""")
subset = ['AdjSalePrice', 'SqFtTotLiving', 'SqFtLot', 'Bathrooms',
          'Bedrooms', 'BldgGrade']
print("house = pd.read_csv('house_sales.csv', sep='\t')")
house = pd.read_csv(common.HOUSE_CSV, sep='\t')
print("print(house[subset].head())")
print(house[subset].head())

print("""
predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms',
              'Bedrooms', 'BldgGrade']
outcome = 'AdjSalePrice'
house_lm = LinearRegression()
house_lm.fit(house[predictors], house[outcome])
print(f'Intercept: {house_lm.intercept_:.3f}')
print('Coefficients:')
for name, coef in zip(predictors, house_lm.coef_):
    print(f' {name}: {coef}')""" )
predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms',
              'Bedrooms', 'BldgGrade']
outcome = 'AdjSalePrice'
house_lm = LinearRegression()
house_lm.fit(house[predictors], house[outcome])
print(f'Intercept: {house_lm.intercept_:.3f}')
print('Coefficients:')
for name, coef in zip(predictors, house_lm.coef_):
    print(f' {name}: {coef}')
print()

fitted = house_lm.predict(house[predictors])
RMSE = np.sqrt(mean_squared_error(house[outcome], fitted))
r2 = r2_score(house[outcome], fitted)
print(f'RMSE: {RMSE:.0f}')
print(f'r2: {r2:.4f}')

print( """
  # While scikit-learn provides a variety of different metrics,
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
