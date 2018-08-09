using TimeSeries
using FredData
using Gadfly

f = Fred("2feb6269f18df39acfd321ae549b7de3")

"""
FUNCTION DEFINITIONS
"""

function get_vintage(srs, date)
    df = srs.data # might be easier with Query
    mask = df[:realtime_start] .<= date .< df[:realtime_end]
    temp = df[mask,:]
    ts = TimeArray(temp[:date].data, temp[:value].data, [srs.id])
end

function transform(ts::TimeArray, how)
    if how == :unchanged
        ts = ts
    elseif how == :log
        ts = TimeArray(ts.timestamp, log.(ts.values), ts.colnames)
    elseif how == :Δ
        ts = TimeArray(ts.timestamp[2:end], ts.values[2:end,:].-ts.values[1:end-1,:], ts.colnames)
    elseif how == :Δlog
        ts = TimeArray(ts.timestamp[2:end], log.(ts.values[2:end,:]).-log.(ts.values[1:end-1,:]), ts.colnames)
    elseif how == :Δ2log
        vals = log.(ts.values[2:end,:]).-log.(ts.values[1:end-1,:])
        ts = TimeArray(ts.timestamp[3:end], vals[2:end,:].-vals[1:end-1,:], ts.colnames)
    else
        throw(error)
    end
    return ts
end

function merge_timeseries(array)
    merged_ts = merge(array[1], array[2], :outer)
    for ts=array[3:end]
         merged_ts = merge(merged_ts, ts, :outer)
    end
    return merged_ts
end

"""
PART I: GDP
"""

vintage_date = Date(2017,12,15)

GDP = get_data(f, "GDPC1"; realtime_start="1999-12-31")
gdp = get_vintage(GDP, vintage_date)
gdp = TimeArray(gdp.timestamp.+Dates.Month(2), gdp["GDPC1"].values, gdp.colnames) # so that quarterly value is assigned to last month
gdp = transform(gdp, :Δlog)
gdp = from(gdp, Date(1990, 1, 1))

"""
PART II: Monthly predictors
"""

# TODO: expand set of predictors

# transformations as suggested in FRED-MD
predictors1 = Dict(
    "INDPRO"   => :Δlog,      # Industrial Production Index
    "TCU"      => :Δ,         # Capacity Utilization: Total Industry
    "UNRATE"   => :Δ,         #
    "PAYEMS"   => :Δlog,      # All Employees: Total Nonfarm Payrolls
    "CPIAUCSL" => :Δ2log,     #
    "PPIACO"   => :Δ2log,     #
    "HOUST"    => :log,       #
    "PERMIT"   => :log,       #
    "FEDFUNDS" => :Δ
)

# transformations that I find it appropriate
predictors2 = Dict(
    "INDPRO"   => :Δlog,      # Industrial Production Index
    "TCU"      => :Δ,         # Capacity Utilization: Total Industry
    "UNRATE"   => :Δ,         #
    "PAYEMS"   => :Δlog,      # All Employees: Total Nonfarm Payrolls
    "CPIAUCSL" => :Δlog,     #
    "PPIACO"   => :Δlog,     #
    "HOUST"    => :Δlog,       #
    "PERMIT"   => :Δlog,       #
)

predictors = predictors2

# TODO: unit root test ?

dataset = Dict()
for id in keys(predictors)
    dataset[id] = get_data(f, id; realtime_start="1999-12-31")
end

"""
PART III: Joint dataset
"""

data = []

for id in keys(predictors)
    new_series = get_vintage(dataset[id], vintage_date)
    new_series = transform(new_series, predictors[id])
    new_series = from(new_series, Date(1990, 1, 1))
    data = push!(data, new_series)
end

data = unshift!(data, gdp)

data = merge_timeseries(data)

"""
PART IV: Plots
"""

plot(x=gdp.timestamp, y=gdp["GDPC1"].values, Geom.line())

plot(x=data.timestamp, y=data["HOUST"].values, Geom.line)
