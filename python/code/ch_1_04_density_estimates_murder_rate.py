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

print("""  Density Estimates
  Density is an alternative to histograms that can provide more insight into the distribution of the data points.
  Use the argument bw_method to control the smoothness of the density curve (bw_method = 0.3).
""")
print("state = pd.read_csv('state.csv')")
state = pd.read_csv(common.STATE_CSV)

ax = common.printx( "ax = ", """state['Murder.Rate'].plot.hist(density=True, xlim=[0, 12],
                               bins=range(1,12), figsize=(4, 4))""", {'state': state} )
print("""state['Murder.Rate'].plot.density(ax=ax, bw_method=0.3)
ax.set_xlabel('Murder Rate (per 100,000)')
plt.tight_layout()
plt.show()""")

state['Murder.Rate'].plot.density(ax=ax, bw_method=0.3)
ax.set_xlabel('Murder Rate (per 100,000)')
plt.tight_layout()
plt.show()
