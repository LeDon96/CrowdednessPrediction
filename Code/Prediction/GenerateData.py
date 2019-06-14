import pandas as pd
import numpy as np 
from sklearn.metrics.pairwise import rbf_kernel

def TransformDate(date):
    """
    This function derives the weekday number and circular time from the date

    Parameters:
    - date (Timestamp)

    Returns:
    - weekday (int): Number of day of the week
    - is_weekend (int): whether the weekday is in the weekend
    - time (dict): date in circular format
    """

    #Retrieve number of day of the week
    weekday = date.weekday()

    #Check whether it is weekend or not
    if weekday == 5 or weekday == 6:
        is_weekend = 1
    else:
        is_weekend = 0

    #Transform date to circular form
    time = TransformTime(date)

    return weekday, is_weekend, time


def TransformTime(date):
    """
    This function returns the circular time from a date

    Parameters:
    - date (Timestamp): data that needs to be converted

    Returns:
    - Dict with
        - Month: cos and sin of month
        - Day: cos and sin of day
        - Hour: cos and sin of hour
    """

    #Formula to make month circular with cos and sin
    month_sin = np.sin(2 * np.pi * date.month / 12)
    month_cos = np.cos(2 * np.pi * date.month / 12)

    #Formula to make day circular with cos and sin
    day_sin = np.sin(2 * np.pi * date.day / 365)
    day_cos = np.cos(2 * np.pi * date.day / 365)

    #Lists to save ciruclar hours of a single day
    hour_sin = []
    hour_cos = []
    hour_list = []

    #Loop over all given hours and return the cos and sin for each
    for hour in range(100, 2401, 100):
        hour_sin.append(np.sin(2 * np.pi * hour / 2400))
        hour_cos.append(np.cos(2 * np.pi * hour / 2400))
        
        #Save the original hour seperate
        hour_list.append(hour)

    return {"Month Sin": month_sin, "Month Cos": month_cos, "Day Sin": day_sin,
            "Day Cos": day_cos, "Hour Sin": hour_sin, "Hour Cos": hour_cos, "Hour": hour_list}


def SelectSensor(sensor,hour, weekday, stations, sensor_dict, lat_scaler, lon_scaler, full_df, date):
    """
    This function returns the scaled coordinates of the given sensors and the weights and scores of the given stations,
    in relation to eah of the given sensors

    Parameters:
    - hour (int): given hour of the day
    - weekday (int): given day of the week 
    - sensor (str): given sensor
    - stations (list): list of all relevant stations
    - sensor_dict (dict): all latitude and longitude data of each given sensor
    - station_dict (dict): weights and scores of each given station
    - lat_scaler (model): trained scaler to transform given latitude
    - lon_scaler (model): trained scaler to transform given longitude
    - station_scaler (model): trained scaler transform the stations weights and scores
    - passenger_df (df): average passenger counts

    Returns:
    - lon_scaled (float): scaled sensor longitude
    - lat_scaled (float): scaled sensor latitude
    - weights_dict (dict): station weights and scores, in relation to given sensor longitude and latitude
    """

    #Scale the sensor longitude and latitude
    lon_scaled = lon_scaler.transform(
        sensor_dict["Longitude"].reshape(1, -1))
    lat_scaled = lat_scaler.transform(
        sensor_dict["Latitude"].reshape(1, -1))

    #Save the unscaled sensor longitude and latitude in array
    y = np.array((lat_scaled, lon_scaled)).reshape(-1, 1)
    
    #Dict to save station data in
    weights_dict = {}
    coor_dict = {}

    full_df["Date"] = pd.to_datetime(
        full_df["Date"], format="%Y-%m-%d")
    
    df = full_df[(full_df["Date"] == date) & (full_df["Hour"] == hour) & (full_df["Sensor"] == sensor)].reset_index()

    #Loop over al given stations
    for station in stations:
        #Save te average passenger counts of given station
        passengers = df[station + " passengers"][0]

        #Save unscaled station longitude and latitude in array
        x = np.array(df[station + " LatScaled"][0],
                        df[station + " LonScaled"][0]).reshape(1, -1)
        coor_dict[station + " LatScaled"] = df[station + " LatScaled"][0]
        coor_dict[station + " LonScaled"] = df[station + " LonScaled"][0]

        #Calculate rbf kernel between sensor and station longitude and latitude
        weight = rbf_kernel(x, y)

        #Save the station weight and score in dict
        weights_dict.update({station + " score": (weight[0][0] * passengers + weight[0][1] * passengers)/2,
                            station + " weight": (weight[0][0] + weight[0][1])/2})

    return lon_scaled, lat_scaled, weights_dict, coor_dict


def constructSensorData(j, input_dict, date, sensor, sensor_dict, stations, lat_scaler, lon_scaler, full_df):
    """
    This function returns all features needed for a prediction, given a singel sensor and date

    Parameters:
    - j (int): current iteration
    - input_dict (dict): dict with data for all given sensors and dates
    - date (Timestamp): given date
    - sensor (str): given sensor
    - sensor_dict (dict): all latitude and longitude data of each given sensor
    - station_dict (dict): weights and scores of each given station
    - stations (list): list of all relevant stations
    - lat_scaler (model): trained scaler to transform given latitude
    - lon_scaler (model): trained scaler to transform given longitude
    - station_scaler (model): trained scaler transform the stations weights and scores
    - passenger_df (df): average passenger counts

    Returns:
    - j (int): updates iteration
    - input_dict (dict): updated dict with data for all given sensors and dates
    """

    #Retrieve weekday number, whether the day is a weekend day, and the circular time of the given date
    weekday, is_weekend, time = TransformDate(date)

    #Loop over all hours in day and save the needed features in dict
    for i in range(len(time["Hour"])):
        #Retrieve scaled sensor longitude and latitude, and all the weights and scores of the given stations
        sensor_lon, sensor_lat, weights_dict, coor_dict = SelectSensor(sensor, time["Hour"][i], weekday, stations, sensor_dict,
                                                            lat_scaler, lon_scaler, full_df, date)

        input_dict[j] = {"weekday": weekday, "is_weekend": is_weekend, "is_event": 0.0, 
                         "month_sin": time["Month Sin"], "month_cos": time["Month Cos"], 
                         "day_sin": time["Day Sin"],"day_cos": time["Day Cos"], "hour_sin": time["Hour Sin"][i],
                         "hour_cos": time["Hour Cos"][i]}

        input_dict[j].update(weights_dict)

        input_dict[j].update({"LatScaled": sensor_lat, "LonScaled": sensor_lon,
                              "hour": time["Hour"][i],
                              "Sensor": sensor, "Date": date, "SensorLongitude": sensor_dict["Longitude"],
                              "SensorLatitude": sensor_dict["Latitude"]})
        
        input_dict[j].update(coor_dict)

        j += 1

    return j, input_dict


def combineData(dates, sensor, sensor_dict, stations, lat_scaler, lon_scaler, full_df):
    """
    This function constructs dict with all needed data to generate needed predictions

    Parameters:
    - date (Timestamp): given dates
    - sensor (str): given sensors
    - sensor_dict (dict): all latitude and longitude data of each given sensor
    - station_dict (dict): weights and scores of each given station
    - stations (list): list of all relevant stations
    - lat_scaler (model): trained scaler to transform given latitude
    - lon_scaler (model): trained scaler to transform given longitude
    - station_scaler (model): trained scaler transform the stations weights and scores
    - passenger_df (df): average passenger counts

    Returns:
    - df with all needed data to generate prediction
    """

    #Dict where all needed data will be saved
    input_dict = {}
    j = 0

    #Check the size of the given sensors and dates and generate the appropriate data
    for date in dates:
        j, input_dict = constructSensorData(j, input_dict, date, sensor, sensor_dict, stations, lat_scaler,
                                            lon_scaler, full_df)

    return pd.DataFrame.from_dict(input_dict, orient="index")


def generateDates(start_date, end_date):
    """
    This function generates all possible dates between a start and end date

    Parameters:
    - start_date (Timestamp): start date for predictions
    - end_date (Timestamp): end date for predictions

    Returns:
    - dates (list): all possible dates between start and end date
    """

    #List to save all dates in
    dates = []

    #Difference in days between start and end date
    delta = end_date - start_date

    #Loop over total number of days between start and end date
    for i in range(delta.days):
        
        #Save the date, which is i dates from the start date
        dates.append(start_date + pd.Timedelta(i, unit="D"))

    return dates

def defineCoordinates(add_sensors, full_df):
    """
    This function saves all needed coordinates for predicion

    Parameters:
    - sensors (list): all given sensors
    - stations (list): all given stations
    - add_sensors (boolean): whether to add sensors to prediction
    - extra_cor (boolean): whether to add custom sensor to prediction
        - extra_lon (float): Longitude custom sensor
        - extra_lat (float): Latitude custom sensor
    - full_df: full dateset

    Returns:
    - sensor_dict (dict): Longitude and Latitude each given sensor
    - station_dict (dict): Longitude and Latitude each given station
    """

    sensor_dict = {"Longitude": full_df[full_df["Sensor"] == add_sensors].reset_index()["SensorLongitude"][0],
                                "Latitude": full_df[full_df["Sensor"] == add_sensors].reset_index()["SensorLatitude"][0]}

    return sensor_dict
