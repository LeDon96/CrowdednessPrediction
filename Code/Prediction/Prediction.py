import pandas as pd
import numpy as np
import random

import Code.Prediction.GenerateData as pg 
import Code.Prediction.importModels as im 
import Code.Prediction.plotTimeSeries as timeSeries 


def generatePredictions(sensors, model, stations, lat_scaler, lon_scaler, full_df, station_scaler, xgbr_model,
                        passenger_df, output_dict, pred_dict):

    predict_dict = {}

    sensor_dict, station_dict = pg.defineCoordinates(sensors, stations, pred_dict["add_sensors"], pred_dict["extra_coordinate"],
                                                     pred_dict["extra_lon"], pred_dict["extra_lat"], full_df)

    dates = pg.generateDates(pd.to_datetime(
        pred_dict["start_date"]), pd.to_datetime(pred_dict["end_date"]))
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
        timeSeries.plotTimeSeries(series_df.drop(columns={"Date"}), date, output_dict)

    return predict_df

def prediction(output_dict, params_dict, pred_dict, pbar, i):

    full_df = pd.read_csv(output_dict["full_df"])
    passenger_df = pd.read_csv(output_dict["average_passenger_counts"])
    stations = params_dict["stations"]

    if pred_dict["add_sensors"] == False and pred_dict["extra_coordinate"] == False:
        print("At least, either the custom coordinates or the sensors have to be added")
    elif (params_dict["lon_min"] > pred_dict["extra_lon"]) or (params_dict["lon_max"] < pred_dict["extra_lon"]):
        print("Custom Longitude is outside borders")
    elif (params_dict["lat_min"] > pred_dict["extra_lat"]) or (params_dict["lat_max"] < pred_dict["extra_lat"]):
        print("Custom Latitude is outside borders")
    else:
        if pred_dict["add_sensors"] == True:
            sensors = full_df["Sensor"].unique()

            if pred_dict["extra_coordinate"] == True:
                sensors = np.append("Custom", sensors)
        else:
            sensors = np.array("Custom")

        pbar.update(i+1)

        model, lat_scaler, lon_scaler, station_scaler, xgbr_model = im.importModels(
            pred_dict["model"], output_dict)

        pbar.update(i+1)

        df = generatePredictions(sensors, model, stations, lat_scaler, lon_scaler, full_df, station_scaler, 
                                xgbr_model, passenger_df, output_dict, pred_dict)

        pbar.update(i+1)

        df.to_csv(output_dict["Predictions"], index=False)

        pbar.update(i+1)
