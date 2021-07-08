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
predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms',
              'BldgGrade', 'PropertyType', 'ZipGroup']
X = common.printx("X = ", "pd.get_dummies(house[predictors], drop_first=True)",
                          {'pd':pd, 'house':house, 'predictors':predictors} )

confounding_lm = LinearRegression()
confounding_lm.fit(X, house[outcome])
house_98105 = common.printx("house_98105 = ", "house.loc[house['ZipCode'] == 98105, ]",
                                             {'house':house} )
predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms',
              'BldgGrade']
house_outlier = common.printx("house_outlier = ", "sm.OLS(house_98105[outcome], house_98105[predictors].assign(const=1))",
                             {'sm':sm, 'house_98105':house_98105, 'outcome':outcome, 'predictors':predictors} )

print("""result_98105 = house_outlier.fit()""")
result_98105 = house_outlier.fit()

print( "  # The package statsmodel provides a number of plots to analyze the data point influence" )
print("""
influence = OLSInfluence(result_98105)
fig, ax = plt.subplots(figsize=(5, 5))
ax.axhline(-2.5, linestyle='--', color='C1')
ax.axhline(2.5, linestyle='--', color='C1')
ax.scatter(influence.hat_matrix_diag, influence.resid_studentized_internal,
           s=1000 * np.sqrt(influence.cooks_distance[0]),
           alpha=0.5)
ax.set_xlabel('hat values')
ax.set_ylabel('studentized residuals')
plt.tight_layout()
plt.show()""")
influence = OLSInfluence(result_98105)
fig, ax = plt.subplots(figsize=(5, 5))
ax.axhline(-2.5, linestyle='--', color='C1')
ax.axhline(2.5, linestyle='--', color='C1')
ax.scatter(influence.hat_matrix_diag, influence.resid_studentized_internal,
           s=1000 * np.sqrt(influence.cooks_distance[0]),
           alpha=0.5)
ax.set_xlabel('hat values')
ax.set_ylabel('studentized residuals')
plt.tight_layout()
plt.show()

mask = [dist < .08 for dist in influence.cooks_distance[0]]
house_infl = house_98105.loc[mask]
ols_infl = sm.OLS(house_infl[outcome], house_infl[predictors].assign(const=1))
result_infl = ols_infl.fit()

pd.DataFrame({
    'Original': result_98105.params,
    'Influential removed': result_infl.params,
})

print( "  ### Heteroskedasticity, Non-Normality and Correlated Errors" )
print( "  # The 'regplot' in 'seaborn' allows adding a lowess smoothing line to the scatterplot." )

fig, ax = common.printx("fig, ax = ", "plt.subplots(figsize=(5, 5))", {'plt':plt} )
print("""
sns.regplot(x=result_98105.fittedvalues, y=np.abs(result_98105.resid),
            scatter_kws={'alpha': 0.25},
            line_kws={'color': 'C1'},
            lowess=True, ax=ax)
ax.set_xlabel('predicted')
ax.set_ylabel('abs(residual)')
plt.tight_layout()
plt.show()""")
sns.regplot(x=result_98105.fittedvalues, y=np.abs(result_98105.resid),
            scatter_kws={'alpha': 0.25},
            line_kws={'color': 'C1'},
            lowess=True, ax=ax)
ax.set_xlabel('predicted')
ax.set_ylabel('abs(residual)')
plt.tight_layout()
plt.show()

print("""
fig, ax = plt.subplots(figsize=(4, 4))
pd.Series(influence.resid_studentized_internal).hist(ax=ax)
ax.set_xlabel('std. residual')
ax.set_ylabel('Frequency')
plt.tight_layout()
plt.show()""")
fig, ax = plt.subplots(figsize=(4, 4))
pd.Series(influence.resid_studentized_internal).hist(ax=ax)
ax.set_xlabel('std. residual')
ax.set_ylabel('Frequency')
plt.tight_layout()
plt.show()

print( "  ### Partial Residual Plots and Nonlinearity" )
print("""
fig, ax = plt.subplots(figsize=(5, 5))
fig = sm.graphics.plot_ccpr(result_98105, 'SqFtTotLiving', ax=ax)
plt.tight_layout()
plt.show()""")
fig, ax = plt.subplots(figsize=(5, 5))
fig = sm.graphics.plot_ccpr(result_98105, 'SqFtTotLiving', ax=ax)
plt.tight_layout()
plt.show()

print("""
fig = plt.figure(figsize=(8, 12))
fig = sm.graphics.plot_ccpr_grid(result_98105, fig=fig)""")
fig = plt.figure(figsize=(8, 12))
fig = sm.graphics.plot_ccpr_grid(result_98105, fig=fig)

print( "  ### Polynomial and Spline Regression" )

model_poly = common.printx("model_poly = ", """smf.ols(formula='AdjSalePrice ~  SqFtTotLiving + np.power(SqFtTotLiving, 2) + ' +
                                            'SqFtLot + Bathrooms + Bedrooms + BldgGrade', data=house_98105)""",
                                            {'smf':smf, 'np':np, 'house_98105': house_98105} )
print("""result_poly = model_poly.fit()
print(result_poly.summary())""")
result_poly = model_poly.fit()
print(result_poly.summary())

# Next line 434 partial residual plot works only for linear term
