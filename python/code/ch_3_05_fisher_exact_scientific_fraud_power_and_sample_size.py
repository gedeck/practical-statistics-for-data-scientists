#!/usr/local/bin/python

## Practical Statistics for Data Scientists (Python)
## Chapter 3. Statistial Experiments and Significance Testing
# > (c) 2019 Peter C. Bruce, Andrew Bruce, Peter Gedeck

from pathlib import Path
import random
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats import power
import matplotlib.pylab as plt
import common

print("""
  ### Fisher's Exact Test
  Scipy has only an implementation of Fisher's Exact test for 2x2 matrices.
  There is a github repository that provides a Python implementation that uses the
  same code as the R version. Installing this requires a Fortran compiler.

 stats.fisher_exact(clicks)
 stats.fisher_exact(clicks.values)
""")

print("  #### Scientific Fraud (?)")

print("imanishi = pd.read_csv('imanishi_data.csv')")
imanishi = pd.read_csv(common.IMANISHI_CSV)
print("""imanishi.columns = [c.strip() for c in imanishi.columns]
ax = imanishi.plot.bar(x='Digit', y=['Frequency'], legend=False,
                      figsize=(4, 4))
ax.set_xlabel('Digit')
ax.set_ylabel('Frequency')
plt.tight_layout()
plt.show()
""")
imanishi.columns = [c.strip() for c in imanishi.columns]
ax = imanishi.plot.bar(x='Digit', y=['Frequency'], legend=False,
                      figsize=(4, 4))
ax.set_xlabel('Digit')
ax.set_ylabel('Frequency')
plt.tight_layout()
print("plt.show()")
plt.show()

print("""
  # Power and Sample Size
  # statsmodels has a number of methods for power calculation
  # see e.g.: https://machinelearningmastery.com/statistical-power-and-power-analysis-in-python/
""")
print("""effect_size = sm.stats.proportion_effectsize(0.0121, 0.011)
analysis = sm.stats.TTestIndPower()
result = analysis.solve_power(effect_size=effect_size,
                              alpha=0.05, power=0.8, alternative='larger')
print('Sample Size: %.3f' % result)
""")
effect_size = sm.stats.proportion_effectsize(0.0121, 0.011)
analysis = sm.stats.TTestIndPower()
result = analysis.solve_power(effect_size=effect_size,
                              alpha=0.05, power=0.8, alternative='larger')
print('Sample Size: %.3f' % result)

print("""
effect_size = sm.stats.proportion_effectsize(0.0165, 0.011)
analysis = sm.stats.TTestIndPower()
result = analysis.solve_power(effect_size=effect_size,
                              alpha=0.05, power=0.8, alternative='larger')
print('Sample Size: %.3f' % result)""")
effect_size = sm.stats.proportion_effectsize(0.0165, 0.011)
analysis = sm.stats.TTestIndPower()
result = analysis.solve_power(effect_size=effect_size,
                              alpha=0.05, power=0.8, alternative='larger')
print('Sample Size: %.3f' % result)
