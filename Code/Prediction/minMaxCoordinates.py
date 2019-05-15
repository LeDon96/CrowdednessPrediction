import pandas as pd 

def minMaxCoordinates(df):

    lon_max = df["SensorLongitude"].max()
    lon_min = df["SensorLongitude"].min()
    lat_max = df["SensorLatitude"].max()
    lat_min = df["SensorLatitude"].min()

    return lon_max, lon_min, lat_max, lat_min
