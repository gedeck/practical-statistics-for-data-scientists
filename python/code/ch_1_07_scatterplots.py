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
from matplotlib.collections import EllipseCollection
from matplotlib.colors import Normalize

print("""  Scatterplots
  Simple scatterplots are supported by Pandas. Specifying the marker as $\u25EF$
  uses an open circle for each point.

sp500_sym = pd.read_csv('sp500_sectors.csv')
sp500_px = pd.read_csv('sp500_data.csv.gz', index_col=0)""")

sp500_sym = pd.read_csv(common.SP500_SECTORS_CSV)
sp500_px = pd.read_csv(common.SP500_DATA_CSV, index_col=0)
telecomSymbols = common.printx( "", "sp500_sym[sp500_sym['sector'] == 'telecommunications_services']['symbol']",
                              {'sp500_sym': sp500_sym} )
telecom = common.printx( "telecom = ", "sp500_px.loc[sp500_px.index >= '2012-07-01', telecomSymbols]",
                                      {'sp500_px': sp500_px, 'telecomSymbols': telecomSymbols}  )
common.printx( "",  "telecom.corr()", {'telecom': telecom} )

ax = common.printx( "ax = ", "telecom.plot.scatter(x='T', y='VZ', figsize=(4, 4), marker='$\u25EF$')",
                  {'telecom': telecom} )
ax.set_xlabel('ATT (T)')
ax.set_ylabel('Verizon (VZ)')
ax.axhline(0, color='grey', lw=1)
ax.axvline(0, color='grey', lw=1)

plt.tight_layout()
plt.show()

ax = telecom.plot.scatter(x='T', y='VZ', figsize=(4, 4), marker='$\u25EF$', alpha=0.5)
ax.set_xlabel('ATT (T)')
ax.set_ylabel('Verizon (VZ)')
ax.axhline(0, color='grey', lw=1)
print(ax.axvline(0, color='grey', lw=1))

print("""  Exploring Two or More Variables
  Load the King County WA dataset and filter based on a variety of criteria""")

print("kc_tax = pd.read_csv(common.KC_TAX_CSV)")
kc_tax = pd.read_csv(common.KC_TAX_CSV)
kc_tax0 = common.printx( "kc_tax0 = ", """kc_tax.loc[(kc_tax.TaxAssessedValue < 750000) &
                     (kc_tax.SqFtTotLiving > 100) &
                     (kc_tax.SqFtTotLiving < 3500), :]""",
                     {'kc_tax': kc_tax} )
print("print(kc_tax0.shape)")
print(kc_tax0.shape)

print("""  Hexagonal binning and Contours
  Plotting numeric versus numeric data
  If the number of data points gets large, scatter plots will no longer
  be meaningful. Here methods that visualize densities are more useful.
  The hexbin method for pandas data frames is one powerful approach.

  These graphs are very slow to draw.
  """)

ax = common.printx( "ax = ", """kc_tax0.plot.hexbin(x='SqFtTotLiving', y='TaxAssessedValue',
                         gridsize=30, sharex=False, figsize=(5, 4))""", {'kc_tax0': kc_tax0} )
print("ax.set_xlabel('Finished Square Feet')")
ax.set_xlabel('Finished Square Feet')
print("ax.set_ylabel('Tax Assessed Value')")
ax.set_ylabel('Tax Assessed Value')

plt.tight_layout()
plt.show()

print("""  The seaborn kdeplot is a two-dimensional extension of the density plot.""")

print("""fig, ax = plt.subplots(figsize=(4, 4))
sns.kdeplot(data=kc_tax0, x='SqFtTotLiving', y='TaxAssessedValue', ax=ax)
ax.set_xlabel('Finished Square Feet')
ax.set_ylabel('Tax Assessed Value')
plt.tight_layout()
plt.show()""")
fig, ax = plt.subplots(figsize=(4, 4))
sns.kdeplot(data=kc_tax0, x='SqFtTotLiving', y='TaxAssessedValue', ax=ax)
ax.set_xlabel('Finished Square Feet')
ax.set_ylabel('Tax Assessed Value')
plt.tight_layout()
plt.show()
