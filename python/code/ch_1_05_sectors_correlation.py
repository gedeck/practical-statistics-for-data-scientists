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

print("""
sp500_sym = pd.read_csv('sp500_sectors.csv')
sp500_px = pd.read_csv( 'sp500_data.csv.gz', index_col=0)""")
sp500_sym = pd.read_csv(common.SP500_SECTORS_CSV)
sp500_px = pd.read_csv( common.SP500_DATA_CSV, index_col=0)

print()
print( "  # Table 1-7" )
print("""
  Determine telecommunications symbols

telecomSymbols = sp500_sym[sp500_sym['sector'] == 'telecommunications_services']['symbol']
""")
telecomSymbols = sp500_sym[sp500_sym['sector'] == 'telecommunications_services']['symbol']

print("  Filter data for dates July 2012 through June 2015")
telecom = common.printx( "telecom = ", "sp500_px.loc[sp500_px.index >= '2012-07-01', telecomSymbols]",
                        {'sp500_px': sp500_px, 'telecomSymbols': telecomSymbols} )
print("""telecom.corr()
print(telecom)""")
telecom.corr()
print(telecom)
print()

print("  Next we focus on funds traded on major exchanges (sector == 'etf').")

etfs = common.printx( "etfx = ",  """sp500_px.loc[sp500_px.index > '2012-07-01',
                    sp500_sym[sp500_sym['sector'] == 'etf']['symbol']]""",
                    {'sp500_px': sp500_px, 'sp500_sym': sp500_sym}  )
print("print(etfs.head())")
print(etfs.head())

print("""
  Due to the large number of columns in this table, looking at the correlation matrix
  is cumbersome and it's more convenient to plot the correlation as a heatmap.
  The seaborn package provides a convenient implementation for heatmaps.
""")

print("fig, ax = plt.subplots(figsize=(5, 4))")
fig, ax = plt.subplots(figsize=(5, 4))

ax = common.printx( "ax = ", """sns.heatmap(etfs.corr(), vmin=-1, vmax=1,
                         cmap=sns.diverging_palette(20, 220, as_cmap=True),
                         ax=ax)""", {'sns': sns, 'etfs': etfs, 'ax': ax} )
print("""plt.tight_layout()
plt.show()""")
plt.tight_layout()
plt.show()
