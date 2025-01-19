#!/usr/local/bin/python

## Practical Statistics for Data Scientists (Python)
print( "  ## Chapter 2. Data and Sampling Distributions" )
# > (c) 2019 Peter C. Bruce, Andrew Bruce, Peter Gedeck

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pylab as plt
import common

print( "  ## Normal Distribution" )
print("  Standard Normal and QQ-Plots")
print("""  The package scipy has the function 'scipy.stats.probplot' to create QQ-plots.
  The argument 'dist' specifies the distribution, which is set by default to the normal distribution.
""")

print("""fig, ax = plt.subplots(figsize=(4, 4))
norm_sample = stats.norm.rvs(size=100)
stats.probplot(norm_sample, plot=ax)
plt.tight_layout()
plt.show()""")
fig, ax = plt.subplots(figsize=(4, 4))
norm_sample = stats.norm.rvs(size=100)
stats.probplot(norm_sample, plot=ax)
plt.tight_layout()
plt.show()
print()

print( "  ## Long-Tailed Distributions" )
print("""sp500_px = pd.read_csv('sp500_data.csv.gz')
nflx = sp500_px.NFLX
nflx = np.diff(np.log(nflx[nflx>0]))
fig, ax = plt.subplots(figsize=(4, 4))
stats.probplot(nflx, plot=ax)
plt.tight_layout()
plt.show()""")
sp500_px = pd.read_csv(common.SP500_DATA_CSV)
nflx = sp500_px.NFLX
nflx = np.diff(np.log(nflx[nflx>0]))
fig, ax = plt.subplots(figsize=(4, 4))
stats.probplot(nflx, plot=ax)
plt.tight_layout()
plt.show()
print()

print( "  ## Binomial Distribution" )
print("""print(stats.binom.pmf(2, n=5, p=0.1))
print(stats.binom.cdf(2, n=5, p=0.1))""" )
print(stats.binom.pmf(2, n=5, p=0.1))
print(stats.binom.cdf(2, n=5, p=0.1))
print()

print( "  ## Poisson and Related Distribution" )
print( "  ### Poisson Distributions" )
print( """sample = stats.poisson.rvs(2, size=100)
pd.Series(sample).plot.hist()
plt.show()""")
sample = stats.poisson.rvs(2, size=100)
pd.Series(sample).plot.hist()
plt.show()
print()

print( "  ### Exponential Distribution" )
print("""sample = stats.expon.rvs(scale=5, size=100)
pd.Series(sample).plot.hist()
plt.show()""")
sample = stats.expon.rvs(scale=5, size=100)
pd.Series(sample).plot.hist()
plt.show()
print()

print( "  ###  Weibull Distribution" )
print("""sample = stats.weibull_min.rvs(1.5, scale=5000, size=100)
pd.Series(sample).plot.hist()
plt.show()""")
sample = stats.weibull_min.rvs(1.5, scale=5000, size=100)
pd.Series(sample).plot.hist()
plt.show()
