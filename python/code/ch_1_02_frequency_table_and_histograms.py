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

print("""  Put the US states into \"bins\" by population:
  Frequency Table and Histograms
  The `cut` method for pandas data splits the dataset into bins.
  There are a number of arguments for the method.
  The following code creates equal sized bins.
  The method `value_counts` returns a frequency table.
""")

print("state = pd.read_csv('state.csv')")
state = pd.read_csv(common.STATE_CSV)
print("""binnedPopulation = pd.cut(state['Population'], 10)
print(binnedPopulation.value_counts())
""")
binnedPopulation = pd.cut(state['Population'], 10)
print(binnedPopulation.value_counts())

print()
print( "  # Table 1.5" )
print()
print("binnedPopulation.name = 'binnedPopulation'")
binnedPopulation.name = 'binnedPopulation'
df = common.printx( "df = ", "pd.concat([state, binnedPopulation], axis=1)", {'state': state, 'pd': pd, 'binnedPopulation': binnedPopulation})
df = common.printx( "df = ", "df.sort_values(by='Population')", {'df': df})

print()
print("groups = []")
groups = []
print("""
for group, subset in df.groupby(by='binnedPopulation'):
    groups.append({
        'BinRange': group,
        'Count': len(subset),
        'States': ','.join(subset.Abbreviation)
    })
print(pd.DataFrame(groups))""")

for group, subset in df.groupby(by='binnedPopulation'):
    groups.append({
        'BinRange': group,
        'Count': len(subset),
        'States': ','.join(subset.Abbreviation)
    })
print(pd.DataFrame(groups))

print()
print("Pandas also supports histograms for exploratory data analysis.")

ax = common.printx( "ax = ", "(state['Population'] / 1_000_000).plot.hist(figsize=(4, 4))", {'state': state})
print("ax.set_xlabel('Population (millions)')")
ax.set_xlabel('Population (millions)')

print("""plt.tight_layout()
plt.show()""")
plt.tight_layout()
plt.show()
