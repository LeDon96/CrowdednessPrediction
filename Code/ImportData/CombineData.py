#Imports
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

def strToTimestamp(df, format):
    """
    This function converts a pandas.df column to Timestamp object

    Parameters:
    - df (df[col]): Needs to be converted to pd.Timestamp
    - format (str): format of the date as it's given

    Returns: DF[col] with al dates as pd.Timestamps
    """

    return pd.to_datetime(df, format=format)

def startEndDate(df1, df2):
    """
    This function returns the min date of a given df column and the max date of a given df column

    Parameters:
    - df1 (df[col]): From which the min date has to be returned
    - df2 (df[col]) (optional): From which the max date has to be returned
        - Optional: If this parameter is not given, the value of df1 will be used
        - Useful if max and min date are not in the same df[col]
    
    Returns: Min and Max date if given column(s)
    """

    #if one dataframe is defined 
    if df2 is None:
        return df1.min(), df1.max()

    #If two dataframes are defined
    else: 
        return df1.min(), df2.max()

def importData(sensor_df, gvb_df, event_df):
    """
    This function converts the date from str to pd.Timestamp object 
    
    Parameters:
    - sensor_df (df): sensor data
    - gvb_df (df): gvb data
    - event_df (df): event data

    Returns: Returns all Df's with pd.Timestamp objects
    """

    #Variables
    
    #Format Datetime
    date_format = "%Y-%m-%d"

    #################################################################################

    #Loop over all given DF's and transform str to timestamp
    sensor_df["Date"] = strToTimestamp(sensor_df["Date"], date_format)
    gvb_df["Date"] = strToTimestamp(gvb_df["Date"], date_format)
    event_df["Date"] = strToTimestamp(event_df["Date"], date_format)


    return sensor_df, gvb_df, event_df

def changeStartEndDate(sensor_df, gvb_df, event_df):
    """
    This function selects rows of df's based on generated start and end dates
    
    Parameters:
    - sensor_df (df): sensor data
    - gvb_df (df): gvb data
    - event_df (df): event data

    Returns: Returns all DF's within given start and end dates
    """

    #Variables

    #Select start and end date
    start_date, end_date = startEndDate(sensor_df["Date"], gvb_df["Date"])

    #################################################################################

    #Loop over all given DF's and select rows based on start and end date
    sensor_df = sensor_df[(sensor_df["Date"] >= start_date) & (
        sensor_df["Date"] <= end_date)].reset_index().drop(columns=["index"])
    
    gvb_df = gvb_df[(gvb_df["Date"] >= start_date) & (
        gvb_df["Date"] <= end_date)].reset_index().drop(columns=["index"])

    event_df = event_df[(event_df["Date"] >= start_date) & (
        event_df["Date"] <= end_date)].reset_index().drop(columns=["index"])

    return sensor_df, gvb_df, event_df


def calculateWeights(stations, df):
    """
    This function returns a dict with scaled rbk kernels, representing the distance between each station and sensor. 

    Parameters:
    - stations (list): all relevant stations
    - df (df): where the latitudes and longitudes of each station and sensor are stored

    Returns: Dict with all scaled weights per sensor, per station
    """

    #Variables

    #List all sensors present in full dataset
    sensors = df["Sensor"].unique()

    weights = {}

    #################################################################################

    #Loop over all the sensors
    for sensor in sensors:

        #Make an array with the latitude and longitude of the sensor
        y = np.array(df[df["Sensor"] == sensor].reset_index()["Latscaled"][0],
                      df[df["Sensor"] == sensor].reset_index()["Lonscaled"][0]).reshape(-1, 1)

        station_weights = {}
        #Loop over all stations
        for station in stations:

            #Make an array with the latitude and longitude of the station
            x = np.array(df[station + " LatScaled"][0],
                          df[station + " LonScaled"][0]).reshape(1, -1)

            #Add station weight
            station_weights[station + " weight"] = rbf_kernel(x, y)

        weights[sensor] = station_weights

    return weights


def constructFullDF(sensor_df, gvb_df, event_df, stations, lat_scaler_filename, lon_scaler_filename):
    """
    This function combines all the previously constructed DF's and merges them into one. In addition, time is transformed into a cyclic continuous feature.

    Parameters:
    - sensor_df (df): sensor data
    - gvb_df (df): gvb data
    - event_df (df): event data
    - stations (list): all relevant stations
    - station_scaler_filename (str): where the scalar for station weights should be stored

    Returns: Full GVB that contains all relevant data
    """

    latscaler = StandardScaler()
    lonscaler = StandardScaler()

    #Combine DF's
    gvb_sensor_df = pd.merge(gvb_df, sensor_df, on=[
        "Date", "Hour", "weekday"], how="outer")
    full_df = pd.merge(gvb_sensor_df, event_df, on=["Date"], how="outer")

    #################################################################################

    #Sort keys on date
    full_df = full_df.sort_values(
        by=["Date"]).reset_index().drop(columns=["index"])

    #Fill NaN values with 0.0
    full_df = full_df.fillna(0.0)

    #Add columns for the cos and sin of month, day and year
    full_df = full_df.assign(Year=0, month_sin=0, month_cos=0,
                             day_sin=0, day_cos=0, hour_sin=0, hour_cos=0)

    #Train scaler
    lats = []
    lons = []

    lats.append(full_df["SensorLatitude"].values)
    lons.append(full_df["SensorLongitude"].values)

    for station in stations:
        full_df[station + " score"] = 0
        full_df[station + " weight"] = 0
        full_df[station + " passengers"] = 0
        lats.append(full_df[station + " Lat"].values)
        lons.append(full_df[station + " Lon"].values)

    lats = np.asarray(lats).reshape(-1, 1)
    latscaler.fit(lats)

    lons = np.asarray(lons).reshape(-1, 1)
    lonscaler.fit(lons)

    pickle.dump(latscaler, open(lat_scaler_filename, 'wb'))
    pickle.dump(lonscaler, open(lon_scaler_filename, 'wb'))

    full_df["Latscaled"] = latscaler.transform(
        full_df["SensorLatitude"].values.reshape(-1, 1))
    full_df["Lonscaled"] = lonscaler.transform(
        full_df["SensorLongitude"].values.reshape(-1, 1))

    for station in stations:
        full_df[station + " LatScaled"] = latscaler.transform(
            full_df[station + " Lat"].values.reshape(-1, 1))
        full_df[station + " LonScaled"] = lonscaler.transform(
            full_df[station + " Lon"].values.reshape(-1, 1))

    #################################################################################

    #Construct dict with station weigths
    station_weights = calculateWeights(stations, full_df)

    #################################################################################

    #Transform DF to Dict
    time_dict = full_df.to_dict("index")

    #Transform Date to seperate year, month, day and hour. And transform month, day, hour to cos/sin to make it circular
    for k, v in time_dict.items():
        v["Year"] = v["Date"].year

        v["month_sin"] = np.sin(2 * np.pi * v["Date"].month / 12)
        v["month_cos"] = np.cos(2 * np.pi * v["Date"].month / 12)

        v["day_sin"] = np.sin(2 * np.pi * v["Date"].day / 365)
        v["day_cos"] = np.cos(2 * np.pi * v["Date"].day / 365)

        v["hour_sin"] = np.sin(2 * np.pi * v["Hour"] / 2400)
        v["hour_cos"] = np.cos(2 * np.pi * v["Hour"] / 2400)

        #Loop over all stations
        for station in stations:

            #Add a station score, which is the weight multiplied with total passengers
            v[station + " score"] = station_weights[v["Sensor"]][station + " weight"][0][0] * (
                v[station + " Arrivals"] + v[station + " Departures"])

            #Add station weight
            v[station + " weight"] = station_weights[v["Sensor"]][station + " weight"][0][0]

            v[station + " passengers"] = v[station +
                                           " Arrivals"] + v[station + " Departures"]

    #Transform dict back to DF
    full_df = pd.DataFrame.from_dict(
        time_dict, orient="index").reset_index().drop(columns="index")

    #################################################################################

    #Drop nonrelevant columns
    for station in stations:
        full_df.drop(columns={station + " Arrivals",
                              station + " Departures"}, inplace=True)

    return full_df


def fullDF(sensor_df, gvb_df, event_df, stations, lat_scaler_filename, lon_scaler_filename):
    """
    This functions constructs the full DF by combining previously constructed DF's

    Parameters:
    - sensor_df (df): sensor data
    - gvb_df (df): gvb data
    - event_df (df): event data
    - stations (list): all relevant stations
    - coor_scaler_filename (str): where the scalar for coordinate weights should be stored

    Returns: Full DF with all relevant data
    """

    #Import the needed CSV files
    sensor_df, gvb_df, event_df = importData(sensor_df, gvb_df, event_df)

    #Change start and end date of DF's
    sensor_df, gvb_df, event_df = changeStartEndDate(
        sensor_df, gvb_df, event_df)

    #Form full DF
    full_df = constructFullDF(
        sensor_df, gvb_df, event_df, stations, lat_scaler_filename, lon_scaler_filename)

    return full_df
