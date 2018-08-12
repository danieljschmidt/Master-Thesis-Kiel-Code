import time
import os

import xml.etree.ElementTree as ET

import urllib.request as url_request
import urllib.error as url_error

urlopen = url_request.urlopen
HTTPError = url_error.HTTPError

import pandas as pd

import numpy as np # for np.logical_and

import quandl # for PMI download

# the function download_alfred_series and its helper functions fetch_data and 
# parse are modified versions of the corresponding functions in the fredapi 
# package https://github.com/mortada/fredapi

def download_alfred_series(fred_api_key, options):
    
    url = "https://api.stlouisfed.org/fred/series/observations?"
    for key, value in options.items():
        url += key + "=" + value + "&"
    url += "api_key" + "=" + fred_api_key
    
    root = fetch_data(url)

    if root is None:
        raise ValueError('No data exists for series id: ' + options["series_id"])
    data = {}
    i = 0
    for child in root.getchildren():
        val = child.get('value')
        if val == ".":
            val = float('NaN')
        else:
            val = float(val)
        realtime_start = parse(child.get('realtime_start'))
        realtime_end = parse(child.get('realtime_end'))
        date = parse(child.get('date'))

        data[i] = {'realtime_start': realtime_start,
                   'realtime_end': realtime_end,
                   'date': date,
                   'value': val}
        i += 1
    data = pd.DataFrame(data).T
    return data


def fetch_data(url):
    try:
        response = urlopen(url)
        root = ET.fromstring(response.read())
    except HTTPError as exc:
        root = ET.fromstring(exc.read())
        raise ValueError(root.get('message'))
    return root

def parse(date_str, format='%Y-%m-%d'):
    rv = pd.to_datetime(date_str, format=format)
    if hasattr(rv, 'to_pydatetime'):
        rv = rv.to_pydatetime()
    return rv


fred_api_key   = "2feb6269f18df39acfd321ae549b7de3"

quandl_api_key = "ttcD92FPFS-dHdgihLAS"

realtime_start    = "1995-01-01"
realtime_end      = "2018-06-30"
observation_start = "1980-01-01"
observation_end   = "2018-06-30"

# - - - - - - - MACROECONOMIC SERIES WITH REVISIONS - - - - - - - 

options = {
    "realtime_start"     : realtime_start,
    "realtime_end"       : realtime_end,
    "observation_start"  : observation_start,
    "observation_end"    : observation_end
}

realtime_series_ids = {
   "INDPRO"  : "IP",
   "UNRATE"  : "UR",
   "PAYEMS"  : "EMP",
   "AWHI"    : "AWH",
   "CPIAUCSL": "CPI",
   "PPIACO"  : "PPI",
   "HOUST"   : "HS",
   "PERMIT"  : "BP",
   "DSPIC96" : "RDPI"
}

for series_id in realtime_series_ids:
    options["series_id"] = series_id
    new_id = realtime_series_ids[series_id]
    df = download_alfred_series(fred_api_key, options)
    df.columns= ['date', 'realtime_end', 'realtime_start', new_id]
    path = os.path.join("..", "data",  new_id + ".csv")
    df.to_csv(path, index=False)
    time.sleep(1)
    
# --> special case 1: real GDP

# in the ALFRED database, quarterly GDP is assigned to the first month of a quarter
# in my model however, quarterly GDP is asssigned to the last month of a quarter

new_id    = "RGDP"
options["series_id"] = "GDPC1"
df = download_alfred_series(fred_api_key, options)
df.columns= ['date', 'realtime_end', 'realtime_start', new_id]
df.date = df.date + pd.DateOffset(months=2)
path = os.path.join("..", "data", new_id + ".csv")
df.to_csv(path, index=False)
time.sleep(1)
    
# --> special case 2: real retail sales

new_id = "RRS"

# new series
options["series_id"] = "RRSFS"
df1 = download_alfred_series(fred_api_key, options)
df1.columns= ['date', 'realtime_end', 'realtime_start', new_id]

# discontinued series
options["series_id"] = "RSALES"
df2 = download_alfred_series(fred_api_key, options)
df2.columns= ['date', 'realtime_end', 'realtime_start', new_id]

df2.loc[df2.realtime_end > df1.realtime_start.min(), "realtime_end"] = df1.realtime_start.min();
df = pd.concat([df2, df1])

cond = np.logical_and(df2.realtime_end   >=  df1.realtime_start.min(), 
                      df2.realtime_start <= df1.realtime_start.min())
df2_last_vint  = df2.loc[cond,["date", new_id]]
df2_right_tail = df2_last_vint.loc[df2_last_vint.date < df1.date.min(),:]
df2_overlap    = df2_last_vint.loc[df2_last_vint.date == df1.date.min(),new_id]
df2_overlap    = df2_overlap.at[df2_overlap.index[0]]
# TODO ugly
df1_overlap = df1.loc[df1.date == df1.date.min(),:]
for i in df1_overlap.index:
    f = df1_overlap.loc[0,new_id]/df2_overlap
    df_rt = df2_right_tail.copy()
    df_rt[new_id] = f*df_rt[new_id]
    df_rt["realtime_start"] = df1_overlap.loc[i, "realtime_start"]
    df_rt["realtime_end"]   = df1_overlap.loc[i, "realtime_end"]
    df = pd.concat([df_rt, df], sort=True)
    
path = os.path.join("..", "data", new_id + ".csv")
df.to_csv(path, index=False)
time.sleep(1)

# - - - - - - - SURVEY DATA - - - - - - - 

# --> Consumer Sentiment Index

# download from ALFRED as the macroeconomic series above

new_id    = "CS"
options["series_id"] = "UMCSENT"
df = download_alfred_series(fred_api_key, options)
df.columns= ['date', 'realtime_end', 'realtime_start', new_id]
path = os.path.join("..", "data", new_id + ".csv")
df.to_csv(path, index=False)
time.sleep(1)

# --> ISM Purchasing Managers Index

# not available on ALFRED
# therefore, download from quandl #
# the result is a dataframe with index "Date" and column "PMI"
# transform the index "Date" into a column "date"
# construct pseudo real-time dataset as follows:
# - realtime_start = date + 1 month
# - realtime_end   = maximum
# possible revisions (due to seasonal adjustment) are not taken into account

new_id = "PMI"
df = quandl.get("ISM/MAN_PMI", api_key=quandl_api_key)   
df.reset_index(inplace=True)                             
df.rename(index=str, columns={"Date": "date", "PMI": new_id}, inplace=True)
df["realtime_start"] = df.date + pd.DateOffset(months=1)
df["realtime_end"]   = realtime_end
path = os.path.join("..", "data", new_id + ".csv")
df.to_csv(path, index=False)
time.sleep(1)


# --> Current General Activity Index for Philadelphia District

# available on ALFRED but limited real-time history
# therefore, download current vintage
# construct pseudo real-time dataset by using the realease calender from their website
# possible revisions (due to seasonal adjustment) are not taken into account

new_id    = "PHIL"
options = {
    "series_id"          : "GACDFSA066MSFRBPHI",
    "observation_start"  : observation_start,
    "observation_end"    : observation_end,
    "realtime_start"     : realtime_end,      #!
    "realtime_end"       : realtime_end,
}
df = download_alfred_series(fred_api_key, options)
df.columns= ['date', 'realtime_end', 'realtime_start', new_id]
df["realtime_start"] = df.date + pd.DateOffset(days=20)
df["realtime_end"]   = realtime_end
path = os.path.join("..", "data", new_id + ".csv")
df.to_csv(path, index=False)
time.sleep(1)

# - - - - - - - FINANCIAL DATA - - - - - - - 

options = {
    "observation_start"  : observation_start,
    "observation_end"    : observation_end,
    "realtime_start"     : realtime_end,     #!
    "realtime_end"       : realtime_end
}

# general case: series that can be downloaded in monthly frequency

financial_series_monthly_ids = {
    "WTISPLC" : "OIL", 
    "TWEXBMTH": "USD",    
    "FEDFUNDS": "FF", 
    "T10Y3MM" : "SPREAD",
}

for series_id in financial_series_monthly_ids:
    options["series_id"] = series_id
    new_id = financial_series_monthly_ids[series_id]
    df = download_alfred_series(fred_api_key, options)
    df.columns= ['date', 'realtime_end', 'realtime_start', new_id]
    df["realtime_start"] = df.date + pd.DateOffset(months=1) + pd.DateOffset(days=-1)
    df["realtime_end"]   = realtime_end
    path = os.path.join("..", "data", new_id + ".csv")
    df.to_csv(path, index=False)
    time.sleep(1)

# --> special case 1: DTWEXM, i.e. trade-weighted US Dollar (major currencies)

# we ignore the revisions dues to changes to the trade weights
# because there is a only limited real-time data for "DTWEXM"
    
# --> special case 2: VXO, only available at daily frequency
    
# aggregate to monthly frequency using the FRED API

new_id    = "VXO"

options["series_id"] = "VXOCLS"
options["frequency"] = "m"
options["aggregation_method"] = "avg"

df = download_alfred_series(fred_api_key, options)
df.columns= ['date', 'realtime_end', 'realtime_start', new_id]
df["realtime_start"] = df.date + pd.DateOffset(months=1) + pd.DateOffset(days=-1)
df["realtime_end"]   = realtime_end
path = os.path.join("..", "data", new_id + ".csv")
df.to_csv(path, index=False)
time.sleep(1)
    
#  --> special case 3: S&P 500 only available from 2008 on FRED

# manual download from Yahoo Finance
    
new_id = "SP500"

df = pd.read_csv(os.path.join("..", "data_manually_downloaded", "SP500.csv"),
                 index_col="Date", parse_dates=["Date"])
df = df.loc[df.index <= observation_end,:]
df = df.resample("1M").mean()
df.reset_index(inplace=True)
df = df[["Date", "Close"]]
df.Date = df.Date + pd.DateOffset(days=1) + pd.DateOffset(months=-1)           
df.rename(index=str, columns={"Date": "date", "Close": new_id}, inplace=True)
df["realtime_start"] = df.date + pd.DateOffset(months=1) + pd.DateOffset(days=-1)
df["realtime_end"]   = realtime_end
path = os.path.join("..", "data", new_id + ".csv")
df.to_csv(path, index=False)
time.sleep(1)