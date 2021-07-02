#!/usr/local/bin/python

## Practical Statistics for Data Scientists (Python)"
## Chapter 3. Statistial Experiments and Significance Testing"
# > (c) 2019 Peter C. Bruce, Andrew Bruce, Peter Gedeck"

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

print( "  Resampling" )
print("""
session_times = pd.read_csv('web_page_data.csv')
session_times.Time = 100 * session_times.Time""")
session_times = pd.read_csv(common.WEB_PAGE_DATA_CSV)
session_times.Time = 100 * session_times.Time

ax = common.printx( "ax = ", "session_times.boxplot(by='Page', column='Time', figsize=(4, 4))",
                            {'session_times': session_times} )
ax.set_xlabel('')
ax.set_ylabel('Time (in seconds)')
plt.suptitle('')
plt.tight_layout()
plt.show()

print("""
mean_a = session_times[session_times.Page == 'Page A'].Time.mean()
mean_b = session_times[session_times.Page == 'Page B'].Time.mean()
print(mean_b - mean_a)""")
mean_a = session_times[session_times.Page == 'Page A'].Time.mean()
mean_b = session_times[session_times.Page == 'Page B'].Time.mean()
print(mean_b - mean_a)

print("")
print( "  The following code is different to the R version. idx_A and idx_B are reversed." )
print( "  Permutation test example with stickiness" )
print("""
  This function works by sampling without replacement n2 indices
  and assigning them to the B group; the remaining n1 indices
  are assigned to group A. It returns the difference between the 2 means.

def perm_fun(x, nA, nB):
    n = nA + nB
    idx_B = set(random.sample(range(n), nB))
    idx_A = set(range(n)) - idx_B
    return x.loc[idx_B].mean() - x.loc[idx_A].mean()""")
def perm_fun(x, nA, nB):
    n = nA + nB
    idx_B = set(random.sample(range(n), nB))
    idx_A = set(range(n)) - idx_B
    return x.loc[idx_B].mean() - x.loc[idx_A].mean()

nA = common.printx( "nA = ", "session_times[session_times.Page == 'Page A'].shape[0]", {'session_times': session_times} )
nB = common.printx( "nB = ", "session_times[session_times.Page == 'Page B'].shape[0]", {'session_times': session_times} )
print("")
print("print(perm_fun(session_times.Time, nA, nB))")
print(perm_fun(session_times.Time, nA, nB))

random.seed(1)
print("""
random.seed(1)

  Calling perm_fun() 1000 times makes a distribution of differences in the session times
  for page A and B that can be plotted as a histogram""")
perm_diffs = common.printx( "perm_diffs = ", "[perm_fun(session_times.Time, nA, nB) for _ in range(1000)]",
                           {'perm_fun': perm_fun, 'session_times': session_times, 'nA': nA, 'nB': nB} )
fig, ax = common.printx( "fig, ax = ", "plt.subplots(figsize=(5, 5))", {'plt': plt} )
print("""ax.hist(perm_diffs, bins=11, rwidth=0.9)
ax.axvline(x = mean_b - mean_a, color='black', lw=2)
ax.text(50, 190, 'Observed\ndifference', bbox={'facecolor':'white'})
ax.set_xlabel('Session time differences (in seconds)')
ax.set_ylabel('Frequency')
plt.tight_layout()
plt.show()

    The following, from Chapter 3 - Statistial Experiments and Significance Testing.py, produces error
    "TypeError: '>' not supported between instances of 'list' and 'float'"
print(np.mean(perm_diffs > mean_b - mean_a))""")

ax.hist(perm_diffs, bins=11, rwidth=0.9)
ax.axvline(x = mean_b - mean_a, color='black', lw=2)
ax.text(50, 190, 'Observed\ndifference', bbox={'facecolor':'white'})
ax.set_xlabel('Session time differences (in seconds)')
ax.set_ylabel('Frequency')
plt.tight_layout()
plt.show()
# print(np.mean(perm_diffs > mean_b - mean_a))
