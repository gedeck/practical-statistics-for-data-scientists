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
from pygam import LinearGAM, s, l
from pygam.datasets import wage
import seaborn as sns
import matplotlib.pyplot as plt
# If you uncomment the next two lines, 'plt' doesn't work, because... "UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure."
# from dmba import stepwise_selection
# from dmba import AIC_score
import common

print("house = pd.read_csv('house_sales.csv', sep='\\t')")
house = pd.read_csv(common.HOUSE_CSV, sep='\t')
house_lm_factor = LinearRegression()
predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms',
              'Bedrooms', 'BldgGrade']
outcome = 'AdjSalePrice'
house_lm = LinearRegression()
house_lm.fit(house[predictors], house[outcome])

zip_groups = pd.DataFrame([
    *pd.DataFrame({
        'ZipCode': house['ZipCode'],
        'residual' : house[outcome] - house_lm.predict(house[predictors]),
    })
    .groupby(['ZipCode'])
    .apply(lambda x: {
        'ZipCode': x.iloc[0,0],
        'count': len(x),
        'median_residual': x.residual.median()
    })
]).sort_values('median_residual')
zip_groups['cum_count'] = np.cumsum(zip_groups['count'])
zip_groups['ZipGroup'] = pd.qcut(zip_groups['cum_count'], 5, labels=False, retbins=False)
zip_groups.head()

to_join = zip_groups[['ZipCode', 'ZipGroup']].set_index('ZipCode')
house = house.join(to_join, on='ZipCode')
house['ZipGroup'] = house['ZipGroup'].astype('category')
print()

print( "  ### Confounding variables" )

predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms',
              'BldgGrade', 'PropertyType', 'ZipGroup']

X = common.printx("X = ", "pd.get_dummies(house[predictors], drop_first=True)",
                          {'pd':pd, 'house':house, 'predictors':predictors} )

print("""
confounding_lm = LinearRegression()
confounding_lm.fit(X, house[outcome])
print(f'Intercept: {confounding_lm.intercept_:.3f}')
print('Coefficients:')
for name, coef in zip(X.columns, confounding_lm.coef_):
    print(f' {name}: {coef}')""")

confounding_lm = LinearRegression()
confounding_lm.fit(X, house[outcome])
print(f'Intercept: {confounding_lm.intercept_:.3f}')
print('Coefficients:')
for name, coef in zip(X.columns, confounding_lm.coef_):
    print(f' {name}: {coef}')

print( "  ### Interactions and Main Effects" )
print("""
model = smf.ols(formula='AdjSalePrice ~  SqFtTotLiving*ZipGroup + SqFtLot + ' +
     'Bathrooms + Bedrooms + BldgGrade + PropertyType', data=house)
results = model.fit()
print(results.summary())""")
model = smf.ols(formula='AdjSalePrice ~  SqFtTotLiving*ZipGroup + SqFtLot + ' +
     'Bathrooms + Bedrooms + BldgGrade + PropertyType', data=house)
results = model.fit()
print(results.summary())

print( "  # Results differ from R due to different binning. ")
print( "  Enforcing the same binning gives identical results" )
print( "  ## Testing the Assumptions: Regression Diagnostics" )
print( "  ### Outliers" )
print( "  # The _statsmodels_ package has the most developed support for outlier analysis." )

house_98105 = common.printx("house_98105 = ", "house.loc[house['ZipCode'] == 98105, ]",
                                             {'house':house} )
predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms',
              'BldgGrade']
house_outlier = common.printx("house_outlier = ", "sm.OLS(house_98105[outcome], house_98105[predictors].assign(const=1))",
                             {'sm':sm, 'house_98105':house_98105, 'outcome':outcome, 'predictors':predictors} )
print("""result_98105 = house_outlier.fit()
print(result_98105.summary())""")
result_98105 = house_outlier.fit()
print(result_98105.summary())

print( """  # The 'OLSInfluence' class is initialized with the OLS regression results and gives access
  to a number of useful properties. Here we use the studentized residuals.""" )
print("""influence = OLSInfluence(result_98105)
sresiduals = influence.resid_studentized_internal
print(sresiduals.idxmin(), sresiduals.min())
print(result_98105.resid.loc[sresiduals.idxmin()])""")
influence = OLSInfluence(result_98105)
sresiduals = influence.resid_studentized_internal
print(sresiduals.idxmin(), sresiduals.min())
print(result_98105.resid.loc[sresiduals.idxmin()])

outlier = common.printx("outlier = ", "house_98105.loc[sresiduals.idxmin(), :]", {'house_98105':house_98105, 'sresiduals':sresiduals} )
print("""print('AdjSalePrice', outlier[outcome])
print(outlier[predictors])""")
print('AdjSalePrice', outlier[outcome])
print(outlier[predictors])
print()

print( "  ### Influential values graphical plot" )
print("""
from scipy.stats import linregress
np.random.seed(5)
x = np.random.normal(size=25)
y = -x / 5 + np.random.normal(size=25)
x[0] = 8
y[0] = 8""")
from scipy.stats import linregress
np.random.seed(5)
x = np.random.normal(size=25)
y = -x / 5 + np.random.normal(size=25)
x[0] = 8
y[0] = 8
print("""
def abline(slope, intercept, ax):
    \"\"\"Calculate coordinates of a line based on slope and intercept\"\"\"
    x_vals = np.array(ax.get_xlim())
    return (x_vals, intercept + slope * x_vals)""")
def abline(slope, intercept, ax):
    """Calculate coordinates of a line based on slope and intercept"""
    x_vals = np.array(ax.get_xlim())
    return (x_vals, intercept + slope * x_vals)

fig, ax = common.printx("fig, ax = ", "plt.subplots(figsize=(4, 4))", {'plt':plt} )
print("""ax.scatter(x, y)
slope, intercept, _, _, _ = linregress(x, y)
ax.plot(*abline(slope, intercept, ax))
slope, intercept, _, _, _ = linregress(x[1:], y[1:])
ax.plot(*abline(slope, intercept, ax), '--')
ax.set_xlim(-2.5, 8.5)
ax.set_ylim(-2.5, 8.5)
plt.tight_layout()
plt.show()""")
ax.scatter(x, y)
slope, intercept, _, _, _ = linregress(x, y)
ax.plot(*abline(slope, intercept, ax))
slope, intercept, _, _, _ = linregress(x[1:], y[1:])
ax.plot(*abline(slope, intercept, ax), '--')
ax.set_xlim(-2.5, 8.5)
ax.set_ylim(-2.5, 8.5)
plt.tight_layout()
plt.show()

# Next line 365 The package _statsmodel_
