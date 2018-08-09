import time
import os

import xml.etree.ElementTree as ET

import urllib.request as url_request
import urllib.error as url_error

urlopen = url_request.urlopen
HTTPError = url_error.HTTPError

import pandas as pd

# the function download_alfred_series and its helper functions are modified
# versions of the corresponding functions in the fredapi package 
# https://github.com/mortada/fredapi

def download_alfred_series(api_key, options):
    
    url = "https://api.stlouisfed.org/fred/series/observations?"
    for key, value in options.items():
        url += key + "=" + value + "&"
    url += "api_key" + "=" + api_key
    
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


api_key = "2feb6269f18df39acfd321ae549b7de3"

options = {
    "realtime_start"     : "1990-01-01",
    "realtime_end"       : "2018-06-30",
    "observations_start" : "1980-01-01",
    "observations_end"   : "2018-06-30"
}

realtime_series_ids = [
   "GDPC1",
   "TCU",
   "INDPRO",
   "UNRATE",
   "PAYEMS",
   "CPIAUCSL",
   "PPIACO",
   "HOUST",
   "PERMIT"
]

for series_id in realtime_series_ids:
    options["series_id"] = series_id
    df = download_alfred_series(api_key, options)
    df.columns= ['date', 'realtime_end', 'realtime_start', series_id]
    path = os.path.join("data", "realtime", series_id + ".csv")
    df.to_csv(path, index=False)
    time.sleep(1)

other_series_ids = [
    "FEDFUNDS",
    "",
    
]