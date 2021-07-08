#!/usr/local/bin/python

## Practical Statistics for Data Scientists (Python)
## Chapter 1. Exploratory Data Analysis
# > (c) 2019 Peter C. Bruce, Andrew Bruce, Peter Gedeck

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import trim_mean
from statsmodels import robust
import wquantiles
import seaborn as sns
import matplotlib.pylab as plt
import common

print("""  Estimates of Location
  Example: Location Estimates of Population and Murder Rates""")

print( "  # Table 1-2" )
state = pd.read_csv(common.STATE_CSV)
common.printx( "",  "state.head(8)", {'state': state} )

print("""  Compute the mean, trimmed mean, and median for Population.
  For `mean` and `median` we can use the Pandas methods of the data frame.
  The trimmed mean requires the `trim_mean` function in _scipy.stats_.
""")
print("state = pd.read_csv(common.STATE_CSV)")
state = pd.read_csv(common.STATE_CSV)
common.printx( "",  "state['Population'].mean()", {'state': state}  )

common.printx( "",  "trim_mean(state['Population'], 0.1)", {'trim_mean': trim_mean, 'state': state}  )

common.printx( "",  "state['Population'].median()", {'state': state}  )

print("""  Weighted mean is available with numpy.
  For weighted median, we can use the specialised package `wquantiles`
  (https://pypi.org/project/wquantiles/).
""")
common.printx( "",  "state['Murder.Rate'].mean()", {'state': state}  )

common.printx( "",  "np.average(state['Murder.Rate'], weights=state['Population'])", {'np': np, 'state': state}  )

common.printx( "",  "wquantiles.median(state['Murder.Rate'], weights=state['Population'])", {'wquantiles': wquantiles, 'state': state}  )

print("  Estimates of Variability")

print( "  # Table 1-2" )
common.printx( "",  "state.head(8)", {'state': state}  )

print("  Standard deviation")
common.printx( "",  "state['Population'].std()", {'state': state}  )

print("  Interquartile range is calculated as the difference of the 75% and 25% quantile.")
common.printx( "",  "state['Population'].quantile(0.75) - state['Population'].quantile(0.25)", {'state': state}  )

print("  Median absolute deviation from the median can be calculated with a method in statsmodels")

common.printx( "",  "robust.scale.mad(state['Population'])", {'robust': robust, 'state': state}  )
common.printx( "",  "abs(state['Population'] - state['Population'].median()).median() / 0.6744897501960817", {'abs': abs, 'state': state}  )

print("""  Percentiles and Boxplots
  Pandas has the `quantile` method for data frames.""")
common.printx( "",  "state['Murder.Rate'].quantile([0.05, 0.25, 0.5, 0.75, 0.95])", {'state': state}  )

print( "  # Table 1.4" )
print("""
percentages = [0.05, 0.25, 0.5, 0.75, 0.95]
df = pd.DataFrame(state['Murder.Rate'].quantile(percentages))
df.index = [f'{p * 100}%' for p in percentages]""")
percentages = [0.05, 0.25, 0.5, 0.75, 0.95]
df = pd.DataFrame(state['Murder.Rate'].quantile(percentages))
df.index = [f'{p * 100}%' for p in percentages]
print( "df.transpose()" )
df.transpose()

print(""" Pandas provides a number of basic exploratory plots; one of them is boxplots

ax = (state['Population']/1_000_000).plot.box(figsize=(3, 4))
ax.set_ylabel('Population (millions)')
plt.tight_layout()
plt.show()
""")

ax = (state['Population']/1_000_000).plot.box(figsize=(3, 4))
ax.set_ylabel('Population (millions)')
plt.tight_layout()
plt.show()
