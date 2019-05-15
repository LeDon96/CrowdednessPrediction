import pandas as pd
import numpy as np 
from sklearn.metrics.pairwise import rbf_kernel

def TransformDate(date):
    """
    This function derives the weekday number and circular time from the date
    """

    weekday = date.weekday()

    if weekday == 5 or weekday == 6:
        is_weekend = 1
    else:
        is_weekend = 0

    time = TransformTime(date)

    return weekday, is_weekend, time


def TransformTime(date):
    """
    This function returns the circular time from a date
    """

    month_sin = np.sin(2 * np.pi * date.month / 12)
    month_cos = np.cos(2 * np.pi * date.month / 12)

    day_sin = np.sin(2 * np.pi * date.day / 365)
    day_cos = np.cos(2 * np.pi * date.day / 365)

    hour_sin = []
    hour_cos = []
    hour_list = []

    for hour in range(100, 2401, 100):
        hour_sin.append(np.sin(2 * np.pi * hour / 2400))
        hour_cos.append(np.cos(2 * np.pi * hour / 2400))
        hour_list.append(hour)

    return {"Month Sin": month_sin, "Month Cos": month_cos, "Day Sin": day_sin,
            "Day Cos": day_cos, "Hour Sin": hour_sin, "Hour Cos": hour_cos, "Hour": hour_list}


def SelectSensor(weekday, sensor, stations, sensor_dict, station_dict, lat_scaler, lon_scaler, station_scaler, passenger_df):
    """
    This function returns the scaled coordinates of the given sensors and the weights and scores of the given stations,
    in relation to eah of the given sensors
    """

    lon_scaled = lon_scaler.transform(
        sensor_dict[sensor]["Longitude"].reshape(-1, 1))[0, 0]
    lat_scaled = lat_scaler.transform(
        sensor_dict[sensor]["Latitude"].reshape(-1, 1))[0, 0]

    y = np.array(sensor_dict[sensor]["Latitude"],
                 sensor_dict[sensor]["Longitude"]).reshape(1, -1)
    weights_dict = {}

    for station in stations:

        passengers = passenger_df[(passenger_df["Station"] == station) & (
            passenger_df["weekday"] == weekday)].reset_index()["Passengers"][0]

        x = np.array(station_dict[station]["Latitude"],
                     station_dict[station]["Longitude"]).reshape(1, -1)

        weight = rbf_kernel(x, y)[0, 0]

        weights_dict.update({station + " Weight": weight,
                             station + " Score": weight * passengers})

    return lon_scaled, lat_scaled, weights_dict


def constructSensorData(j, input_dict, date, sensor, sensor_dict, station_dict, stations, lat_scaler, lon_scaler, station_scaler, passenger_df):
    """
    This function returns all the features needed for prediciton, given a single sensor and a single date
    """

    weekday, is_weekend, time = TransformDate(date)

    sensor_lon, sensor_lat, weights_dict = SelectSensor(weekday, sensor, stations, sensor_dict,
                                                        station_dict, lat_scaler, lon_scaler, station_scaler, passenger_df)

    for i in range(len(time["Hour Sin"])):
        input_dict[j] = {"weekday": weekday, "is_weekend": is_weekend, "LonScaled": sensor_lon,
                         "LatScaled": sensor_lat, "is_event": 0.0, "month_sin": time["Month Sin"],
                         "month_cos": time["Month Cos"], "day_sin": time["Day Sin"],
                         "day_cos": time["Day Cos"], "hour_sin": time["Hour Sin"][i],
                         "hour_cos": time["Hour Cos"][i], "hour": time["Hour"][i],
                         "Sensor": sensor, "Date": date, "SensorLongitude": sensor_dict[sensor]["Longitude"],
                         "SensorLatitude": sensor_dict[sensor]["Latitude"]}

        input_dict[j].update(weights_dict)

        j += 1

    return j, input_dict


def combineData(dates, sensors, sensor_dict, station_dict, stations, lat_scaler, lon_scaler, station_scaler, passenger_df):
    """
    This function returns all the features needed to generate a prediction
    """

    input_dict = {}
    j = 0

    if sensors.size == 1:
        sensor = np.array2string(sensors).replace("'", "")

        if len(dates) == 1:
            j, input_dict = constructSensorData(j, input_dict, dates, sensor, sensor_dict, station_dict, stations, lat_scaler, lon_scaler,
                                                station_scaler, passenger_df)

        else:

            for date in dates:
                j, input_dict = constructSensorData(j, input_dict, date, sensor, sensor_dict, station_dict, stations, lat_scaler,
                                                    lon_scaler, station_scaler, passenger_df)

    else:
        for sensor in sensors:

            if len(dates) == 1:
                j, input_dict = constructSensorData(j, input_dict, dates, sensor, sensor_dict, station_dict, stations, lat_scaler,
                                                    lon_scaler, station_scaler, passenger_df)

            else:

                for date in dates:
                    j, input_dict = constructSensorData(j, input_dict, date, sensor, sensor_dict, station_dict, stations, lat_scaler,
                                                        lon_scaler, station_scaler, passenger_df)

    return pd.DataFrame.from_dict(input_dict, orient="index")


def generateDates(start_date, end_date):
    """
    This function generates all possible dates between a start and end date
    """

    dates = []
    delta = end_date - start_date

    for i in range(delta.days):
        dates.append(start_date + pd.Timedelta(i, unit="D"))

    return dates
