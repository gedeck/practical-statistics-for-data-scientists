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

print( "  ### F-Statistic" )
print( "  # We can compute an ANOVA table using statsmodel." )

print("""
session_times = pd.read_csv( 'web_page_data.csv' )
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

print("four_sessions = pd.read_csv('four_sessions.csv')")
four_sessions = pd.read_csv(common.FOUR_SESSIONS_CSV)
model = common.printx("model = ", "smf.ols('Time ~ Page', data=four_sessions).fit()",
                                  {'smf': smf, 'four_sessions': four_sessions} )

aov_table = common.printx("aov_table = ", "sm.stats.anova_lm(model)", {'sm': sm, 'model': model} )
print("print(aov_table)")
print(aov_table)

res = common.printx("res = ", """stats.f_oneway(four_sessions[four_sessions.Page == 'Page 1'].Time,
                     four_sessions[four_sessions.Page == 'Page 2'].Time,
                     four_sessions[four_sessions.Page == 'Page 3'].Time,
                     four_sessions[four_sessions.Page == 'Page 4'].Time)""",
                     {'stats': stats, 'four_sessions': four_sessions} )
print(f'F-Statistic: {res.statistic / 2:.4f}')
print(f'p-value: {res.pvalue / 2:.4f}')

print( "  #### Two-way anova only available with statsmodels" )
print( "formula = 'len ~ C(supp) + C(dose) + C(supp):C(dose)'" )
print( "model = ols(formula, data).fit()" )
print( "aov_table = anova_lm(model, typ=2)" )
print( "  ## Chi-Square Test" )
print( "  ### Chi-Square Test: A Resampling Approach" )
print( "  # Table 3-4" )

print("click_rate = pd.read_csv('click_rates.csv')")
click_rate = pd.read_csv(common.CLICK_RATE_CSV)
clicks = common.printx("clicks = ", "click_rate.pivot(index='Click', columns='Headline', values='Rate')",
                                    {'click_rate': click_rate} )
print("print(clicks)")
print(clicks)

print( "  # Table 3-5" )
print("""row_average = clicks.mean(axis=1)
pd.DataFrame({
    'Headline A': row_average,
    'Headline B': row_average,
    'Headline C': row_average,
})""")
row_average = clicks.mean(axis=1)
pd.DataFrame({
    'Headline A': row_average,
    'Headline B': row_average,
    'Headline C': row_average,
})

print( "  # Resampling approach" )
box = [1] * 34
box.extend([0] * 2966)
random.shuffle(box)

def chi2(observed, expected):
    pearson_residuals = []
    for row, expect in zip(observed, expected):
        pearson_residuals.append([(observe - expect) ** 2 / expect
                                  for observe in row])
    # return sum of squares
    return np.sum(pearson_residuals)

expected_clicks = 34 / 3
expected_noclicks = 1000 - expected_clicks
expected = [34 / 3, 1000 - 34 / 3]
chi2observed = chi2(clicks.values, expected)

def perm_fun(box):
    sample_clicks = [sum(random.sample(box, 1000)),
                     sum(random.sample(box, 1000)),
                     sum(random.sample(box, 1000))]
    sample_noclicks = [1000 - n for n in sample_clicks]
    return chi2([sample_clicks, sample_noclicks], expected)

perm_chi2 = [perm_fun(box) for _ in range(2000)]

resampled_p_value = sum(perm_chi2 > chi2observed) / len(perm_chi2)
print(f'Observed chi2: {chi2observed:.4f}')
print(f'Resampled p-value: {resampled_p_value:.4f}')

chisq, pvalue, df, expected = stats.chi2_contingency(clicks)
print(f'Observed chi2: {chisq:.4f}')
print(f'p-value: {pvalue:.4f}')

print( "  ### Figure chi-sq distribution" )

x = [1 + i * (30 - 1) / 99 for i in range(100)]

chi = pd.DataFrame({
    'x': x,
    'chi_1': stats.chi2.pdf(x, df=1),
    'chi_2': stats.chi2.pdf(x, df=2),
    'chi_5': stats.chi2.pdf(x, df=5),
    'chi_10': stats.chi2.pdf(x, df=10),
    'chi_20': stats.chi2.pdf(x, df=20),
})
fig, ax = plt.subplots(figsize=(4, 2.5))
ax.plot(chi.x, chi.chi_1, color='black', linestyle='-', label='1')
ax.plot(chi.x, chi.chi_2, color='black', linestyle=(0, (1, 1)), label='2')
ax.plot(chi.x, chi.chi_5, color='black', linestyle=(0, (2, 1)), label='5')
ax.plot(chi.x, chi.chi_10, color='black', linestyle=(0, (3, 1)), label='10')
ax.plot(chi.x, chi.chi_20, color='black', linestyle=(0, (4, 1)), label='20')
ax.legend(title='df')

plt.tight_layout()
print("plt.show()")
plt.show()
