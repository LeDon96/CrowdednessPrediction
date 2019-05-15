import pandas as pd
import numpy as np
import random

import GenerateData as pg 
import importModels as im 
import minMaxCoordinates as minMax 
import plotTimeSeries as timeSeries 


def generatePredictions(start_date, end_date, sensors, model, stations, lat_scaler, lon_scaler, add_sensors, extra_coordinate,
               extra_lon, extra_lat, full_df, station_scaler, xgbr_model, passenger_df):

    predict_dict = {}

    sensor_dict, station_dict = pg.defineCoordinates(sensors, stations, add_sensors, extra_coordinate,
                                                  extra_lon, extra_lat, full_df)

    dates = pg.generateDates(start_date, end_date)
    df = pg.combineData(dates, sensors, sensor_dict, station_dict,
                     stations, lat_scaler, lon_scaler, station_scaler, passenger_df)
    input_df = df.drop(
        columns={"hour", "Sensor", "Date", "SensorLongitude", "SensorLatitude"}).copy()

    predict_dict["Date"] = df["Date"].copy()
    predict_dict["Hour"] = df["hour"].copy()
    predict_dict["Sensor"] = df["Sensor"].copy()
    predict_dict["SensorLongitude"] = df["SensorLongitude"].copy()
    predict_dict["SensorLatitude"] = df["SensorLatitude"].copy()

    if xgbr_model:
        predict_dict["CrowdednessCount"] = model.predict(
            input_df.values).astype(int)
        predict_dict["CrowdednessCount"][predict_dict["CrowdednessCount"] < 0] = 0
    else:
        predict_dict["CrowdednessCount"] = model.predict(input_df).astype(int)

    predict_df = pd.DataFrame.from_dict(predict_dict)

    for date in dates:

        series_df = predict_df[predict_df["Date"] == date].copy()
        series_df.replace(2400, 0, inplace=True)
        series_df.sort_values(by=["Hour", "Sensor"], inplace=True)
        timeSeries.plotTimeSeries(series_df.drop(columns={"Date"}), date)

    return predict_df

def constructFeatures(model, start_date, end_date, add_sensors, extra_coordinate, extra_lon, extra_lat,
                        full_df, passenger_df, stations):

    if add_sensors == False and extra_coordinate == False:
        print("At least, either the custom coordinates or the sensors have to be added")
    else:

        if add_sensors == True:
            sensors = full_df["Sensor"].unique()

            if extra_coordinate == True:
                sensors = np.append("Custom", sensors)
        else:
            sensors = np.array("Custom")

        model, lat_scaler, lon_scaler, station_scaler, xgbr_model = im.importModels(
            model)

        df = generatePredictions(start_date, end_date, sensors, model, stations, lat_scaler, lon_scaler, add_sensors, extra_coordinate,
                                extra_lon, extra_lat, full_df, station_scaler, xgbr_model, passenger_df)

        return df

def prediction(full_df, passenger_df, stations):

    model = "rfg"
    start_date = pd.to_datetime('2019-10-01')
    end_date = pd.to_datetime('2019-10-05')
    add_sensors = True 
    extra_coordinate = True 

    lon_max, lon_min, lat_max, lat_min = minMax.minMaxCoordinates(full_df)
    extra_lon = round(random.uniform(lon_min, lon_max), 6)
    extra_lat = round(random.uniform(lat_min, lat_max), 6)

    df = constructFeatures(model, start_date, end_date, add_sensors, extra_coordinate, extra_lon, extra_lat,
                           full_df, passenger_df, stations)

    df.to_csv("../../../../Data_thesis/Full_Datasets/Predictions.csv", index=False)

def main():

    full_df = pd.read_csv("../../../../Data_thesis/Full_Datasets/Full.csv")
    passenger_df = pd.read_csv(
        "../../../../Data_thesis/Full_Datasets/AveragePassengerCounts.csv")
    stations = ["Nieuwmarkt", "Nieuwezijds Kolk",
                "Dam", "Spui", "Centraal Station"]

    prediction(full_df, passenger_df, stations)

if __name__ == '__main__':
	main()