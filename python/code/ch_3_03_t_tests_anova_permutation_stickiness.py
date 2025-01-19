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
session_times = pd.read_csv( 'web_page_data.csv' )
session_times.Time = 100 * session_times.Time""")
session_times = pd.read_csv(common.WEB_PAGE_DATA_CSV)
session_times.Time = 100 * session_times.Time

res = common.printx("""  ## t-Tests
res = """, """stats.ttest_ind(session_times[session_times.Page == 'Page A'].Time,
                      session_times[session_times.Page == 'Page B'].Time,
                      equal_var=False)""", {'stats':stats, 'session_times':session_times} )
print(f'p-value for single sided test: {res.pvalue / 2:.4f}')

tstat, pvalue, df = common.printx("tstat, pvalue, df = ", """sm.stats.ttest_ind(
    session_times[session_times.Page == 'Page A'].Time,
    session_times[session_times.Page == 'Page B'].Time,
    usevar='unequal', alternative='smaller')""", {'sm':sm, 'session_times':session_times} )
print(f'p-value: {pvalue:.4f}')
print()

print( "  ## ANOVA" )
print("four_sessions = pd.read_csv('four_sessions.csv')")
four_sessions = pd.read_csv(common.FOUR_SESSIONS_CSV)

ax = four_sessions.boxplot(by='Page', column='Time',
                           figsize=(4, 4))
ax.set_xlabel('Page')
ax.set_ylabel('Time (in seconds)')
plt.suptitle('')
plt.title('')

plt.tight_layout()
print("plt.show()")
plt.show()

print("print(pd.read_csv('four_sessions.csv').head())")
print(pd.read_csv(common.FOUR_SESSIONS_CSV).head())
print()

observed_variance = four_sessions.groupby('Page').mean().var()[0]
print('Observed means:', four_sessions.groupby('Page').mean().values.ravel())
print('Variance:', observed_variance)
print()

print( "  # Permutation test example with stickiness" )
def perm_test(df):
    df = df.copy()
    df['Time'] = np.random.permutation(df['Time'].values)
    return df.groupby('Page').mean().var()[0]

print("print(perm_test(four_sessions))")
print(perm_test(four_sessions))

random.seed(1)
perm_variance = [perm_test(four_sessions) for _ in range(3000)]
print("print('Pr(Prob)', np.mean([var > observed_variance for var in perm_variance]))")
print('Pr(Prob)', np.mean([var > observed_variance for var in perm_variance]))

fig, ax = plt.subplots(figsize=(5, 5))
ax.hist(perm_variance, bins=11, rwidth=0.9)
ax.axvline(x = observed_variance, color='black', lw=2)
ax.text(60, 200, 'Observed\nvariance', bbox={'facecolor':'white'})
ax.set_xlabel('Variance')
ax.set_ylabel('Frequency')

plt.tight_layout()
print("plt.show()")
plt.show()
