#!/usr/local/bin/python

from pathlib import Path

def dataDirectory(dataDirectoryName='data'):
    """
    Return the directory that contains the data.

    We assume that the data folder is locate in a parent directory of this file and named 'data'.
    If your setup is different, you will need to change this method.
    """
    dataDir = Path(__file__).resolve().parent
    while not list(dataDir.rglob('data')):
        dataDir = dataDir.parent
    found = [d for d in dataDir.rglob('data') if d.is_dir()]
    if not found:
        raise Exception(f'Cannot find data directory with name {dataDirectoryName} along the path of your source files')
    return found[0]

try:
    DATA = dataDirectory()
except ImportError:
    DATA = Path().resolve() / 'data'

AIRLINE_STATS_CSV = DATA / 'airline_stats.csv'
KC_TAX_CSV = DATA / 'kc_tax.csv.gz'
LC_LOANS_CSV = DATA / 'lc_loans.csv'
AIRPORT_DELAYS_CSV = DATA / 'dfw_airline.csv'
SP500_DATA_CSV = DATA / 'sp500_data.csv.gz'
SP500_SECTORS_CSV = DATA / 'sp500_sectors.csv'
STATE_CSV = DATA / 'state.csv'
LOANS_INCOME_CSV = DATA / 'loans_income.csv'
WEB_PAGE_DATA_CSV = DATA / 'web_page_data.csv'
FOUR_SESSIONS_CSV = DATA / 'four_sessions.csv'
CLICK_RATE_CSV = DATA / 'click_rates.csv'
IMANISHI_CSV = DATA / 'imanishi_data.csv'
LUNG_CSV = DATA / 'LungDisease.csv'
HOUSE_CSV = DATA / 'house_sales.csv'
LOAN3000_CSV = DATA / 'loan3000.csv'
LOAN_DATA_CSV = DATA / 'loan_data.csv.gz'
FULL_TRAIN_SET_CSV = DATA / 'full_train_set.csv.gz'
LOAN200_CSV = DATA / 'loan200.csv'
HOUSE_TASKS_CSV = DATA / 'housetasks.csv'

def printx(result, expr, vars):
  print( "" )
  print( result + expr )
  return eval( expr, vars )
