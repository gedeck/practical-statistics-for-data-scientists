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

print( "  ## Statistical Significance and P-Values" )
print("""
  The real problem is that people want more meaning from the p-value than it contains.
  Here’s what we would like the p-value to convey:
     The probability that the result is due to chance.
  We hope for a low value, so we can conclude that we’ve proved something. This is
  how many journal editors were interpreting the p-value. But here’s what the p-value
  actually represents:
     The probability that, given a chance model, results as extreme as the observed results could occur.
""")
print("""session_times = pd.read_csv( 'web_page_data.csv' )
session_times.Time = 100 * session_times.Time""")
session_times = pd.read_csv(common.WEB_PAGE_DATA_CSV)
session_times.Time = 100 * session_times.Time
print("""
  This function works by sampling without replacement n2 indices
  and assigning them to the B group; the remaining n1 indices
  are assigned to group A. It returns the difference between the 2 means.

def perm_fun(x, nA, nB):
    n = nA + nB
    idx_B = set(random.sample(range(n), nB))
    idx_A = set(range(n)) - idx_B
    return x.loc[idx_B].mean() - x.loc[idx_A].mean()
""")
def perm_fun(x, nA, nB):
    n = nA + nB
    idx_B = set(random.sample(range(n), nB))
    idx_A = set(range(n)) - idx_B
    return x.loc[idx_B].mean() - x.loc[idx_A].mean()

random.seed(1)
obs_pct_diff = 100 * (200 / 23739 - 182 / 22588)
print(f'Observed difference: {obs_pct_diff:.4f}%')
print("""
conversion = [0] * 45945
conversion.extend([1] * 382)
conversion = pd.Series(conversion)""")
conversion = [0] * 45945
conversion.extend([1] * 382)
conversion = pd.Series(conversion)

perm_diffs = common.printx("perm_diffs = ", """[100 * perm_fun(conversion, 23739, 22588)
              for _ in range(1000)]""", {'perm_fun':perm_fun, 'conversion':conversion} )

fig, ax = plt.subplots(figsize=(5, 5))
ax.hist(perm_diffs, bins=11, rwidth=0.9)
ax.axvline(x=obs_pct_diff, color='black', lw=2)
ax.text(0.06, 200, 'Observed\ndifference', bbox={'facecolor':'white'})
ax.set_xlabel('Conversion rate (percent)')
ax.set_ylabel('Frequency')

plt.tight_layout()
print("plt.show()")
plt.show()
print()

print( "  ### P-Value" )
print( """
  # If 'np.mean' is applied to a list of booleans, it gives the percentage of how often
  # True was found in the list (#True / #Total).""" )
print()
print("print(np.mean([diff > obs_pct_diff for diff in perm_diffs]))")
print(np.mean([diff > obs_pct_diff for diff in perm_diffs]))

print("""survivors = np.array([[200, 23739 - 200], [182, 22588 - 182]])
chi2, p_value, df, _ = stats.chi2_contingency(survivors)
print(f'p-value for single sided test: {p_value / 2:.4f}')""")
survivors = np.array([[200, 23739 - 200], [182, 22588 - 182]])
chi2, p_value, df, _ = stats.chi2_contingency(survivors)
print(f'p-value for single sided test: {p_value / 2:.4f}')
