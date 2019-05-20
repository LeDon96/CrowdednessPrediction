import pandas as pd 

def minMaxCoordinates(df):
    """
    This function generates longitude and latitude borders, based on the min and max values of the sensor 
    longitudes and latitudes

    Parameters:
    - df (df): full dataset

    Returns: 
    - lon_max: Max longitude
    - lon_min: Min longitude
    - lat_max: Max latitude
    - lat_min: Min latitude
    """
    
    lon_max = df["SensorLongitude"].max()
    lon_min = df["SensorLongitude"].min()
    lat_max = df["SensorLatitude"].max()
    lat_min = df["SensorLatitude"].min()

    return lon_max, lon_min, lat_max, lat_min
