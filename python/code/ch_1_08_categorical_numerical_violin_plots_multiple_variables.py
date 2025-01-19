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

print("""  Two Categorical Variables
  The lc_loans dataset
lc_loans = pd.read_csv('lc_loans.csv')
""")
lc_loans = pd.read_csv(common.LC_LOANS_CSV)

print( "  # Table 1-8(1)" )
crosstab = common.printx( "crosstab = ", """lc_loans.pivot_table(index='grade', columns='status',
                                aggfunc=lambda x: len(x), margins=True)""", {'lc_loans': lc_loans} )
print("print(crosstab) ")
print(crosstab)
print()

print( "  # Table 1-8(2)" )
print()
print("""df = crosstab.copy().loc['A':'G',:]
df.loc[:,'Charged Off':'Late'] = df.loc[:,'Charged Off':'Late'].div(df['All'], axis=0)
df['All'] = df['All'] / sum(df['All'])
perc_crosstab = df
print(perc_crosstab)""")
df = crosstab.copy().loc['A':'G',:]
df.loc[:,'Charged Off':'Late'] = df.loc[:,'Charged Off':'Late'].div(df['All'], axis=0)
df['All'] = df['All'] / sum(df['All'])
perc_crosstab = df
print(perc_crosstab)

print("""
  Categorical and Numeric Data
  Pandas boxplots of a column can be grouped by a different column.""")

airline_stats = pd.read_csv(common.AIRLINE_STATS_CSV)
airline_stats.head()
ax = common.printx( "ax = ", "airline_stats.boxplot(by='airline', column='pct_carrier_delay', figsize=(5, 5))",
                   {'airline_stats': airline_stats} )
print("""ax.set_xlabel('')
ax.set_ylabel('Daily % of Delayed Flights')
plt.suptitle('')
plt.tight_layout()
plt.show()""")
ax.set_xlabel('')
ax.set_ylabel('Daily % of Delayed Flights')
plt.suptitle('')
plt.tight_layout()
plt.show()

print()
print("Pandas also supports a variation of boxplots called violinplot.")

fig, ax = common.printx( "fig, ax = ", "plt.subplots(figsize=(5, 5))", {'plt': plt} )
common.printx( "", """sns.violinplot(data=airline_stats, x='airline', y='pct_carrier_delay',
               ax=ax, inner='quartile', color='white')""",
               {'sns': sns, 'ax': ax, 'airline_stats': airline_stats} )
print("""ax.set_xlabel('')
ax.set_ylabel('Daily % of Delayed Flights')
plt.tight_layout()
plt.show()""")
ax.set_xlabel('')
ax.set_ylabel('Daily % of Delayed Flights')
plt.tight_layout()
plt.show()

print("""
  Visualizing Multiple Variables
""")
print("kc_tax = pd.read_csv('kc_tax.csv.gz')")
kc_tax = pd.read_csv(common.KC_TAX_CSV)
kc_tax0 = common.printx( "kc_tax0 = ", """kc_tax.loc[(kc_tax.TaxAssessedValue < 750000) &
                     (kc_tax.SqFtTotLiving > 100) &
                     (kc_tax.SqFtTotLiving < 3500), :]""",
                     {'kc_tax': kc_tax} )
print("print(kc_tax0.shape)")
print(kc_tax0.shape)
print("""zip_codes = [98188, 98105, 98108, 98126]
kc_tax_zip = kc_tax0.loc[kc_tax0.ZipCode.isin(zip_codes),:]""")
zip_codes = [98188, 98105, 98108, 98126]
kc_tax_zip = kc_tax0.loc[kc_tax0.ZipCode.isin(zip_codes),:]

print("""
def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, gridsize=25, cmap=cmap, **kwargs)""")
def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, gridsize=25, cmap=cmap, **kwargs)
g = common.printx( "", "sns.FacetGrid(kc_tax_zip, col='ZipCode', col_wrap=2)",
                  {'sns': sns, 'kc_tax_zip': kc_tax_zip} )
print("""
g.map(hexbin, 'SqFtTotLiving', 'TaxAssessedValue',
      extent=[0, 3500, 0, 700000])
g.set_axis_labels('Finished Square Feet', 'Tax Assessed Value')
g.set_titles('Zip code {col_name:.0f}')
plt.tight_layout()
plt.show()""")
g.map(hexbin, 'SqFtTotLiving', 'TaxAssessedValue',
      extent=[0, 3500, 0, 700000])
g.set_axis_labels('Finished Square Feet', 'Tax Assessed Value')
g.set_titles('Zip code {col_name:.0f}')
plt.tight_layout()
plt.show()
