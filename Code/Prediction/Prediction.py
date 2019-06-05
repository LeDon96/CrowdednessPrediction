import pandas as pd
import numpy as np
import random

import Code.Prediction.GenerateData as pg 
import Code.Prediction.importModels as im 

import matplotlib.pyplot as plt


def generatePredictions(sensors, model, stations, lat_scaler, lon_scaler, station_scaler, full_df, xgbr_model,
                        passenger_df, output_dict, pred_dict, params_dict):
    """
    This function generates crowdedness predictions for specified sensors and dates

    Parameters:
    - sensors (list): all given sensors
    - model (model): desired model to generate predictions with
    - stations (list): all given stations
    - lat_scaler (model): trained scaler to transform given latitude
    - lon_scaler (model): trained scaler to transform given longitude
    - station_scaler (model): trained scaler transform the stations weights and scores
    - full_df (df): full dataset
    - xgbr_model (boolean): check whether model == xgbr
    - passenger_df (df): average passenger counts
    - output_dict (dict): all paths of output files
    - pred_dict (dict): hyperparameters prediction

    Returns:
    - Df with all crowdedness predictions and input prediction data
    """

    #Dict to save all data in
    predict_dict = {}

    #Construct dicts with longitude and latitude given sensors and stations
    sensor_dict, station_dict = pg.defineCoordinates(sensors, stations, pred_dict["add_sensors"], pred_dict["extra_coordinate"],
                                                     pred_dict["extra_lon"], pred_dict["extra_lat"], full_df)

    #Generate all possible dates between start and end date
    dates = pg.generateDates(pd.to_datetime(
        pred_dict["start_date"]), pd.to_datetime(pred_dict["end_date"]))
    
    #Construct df with all needed input data to generate predictions
    df = pg.combineData(dates, sensors, sensor_dict, station_dict,
                     stations, lat_scaler, lon_scaler, station_scaler, passenger_df)
    
    #Remove features that are not needed for the model to generate predictions
    input_df = df.drop(
        columns={"hour", "Sensor", "Date", "SensorLongitude", "SensorLatitude"}).copy()

    #Save needed features for visualization and understanding
    predict_dict["Date"] = df["Date"].copy()
    predict_dict["Hour"] = df["hour"].copy()
    predict_dict["Sensor"] = df["Sensor"].copy()
    predict_dict["SensorLongitude"] = df["SensorLongitude"].copy()
    predict_dict["SensorLatitude"] = df["SensorLatitude"].copy()

    #Generate crowdedness predictions
    if xgbr_model:
        predict_dict["CrowdednessCount"] = model.predict(
            input_df.values)
        predict_dict["CrowdednessCount"][predict_dict["CrowdednessCount"] < 0] = 0
    else:
        predict_dict["CrowdednessCount"] = model.predict(input_df)

    #Convert dict to DF
    predict_df = pd.DataFrame.from_dict(predict_dict)

    #Generate visualizations prediction
    if pred_dict["make_plot"]:
        plt.figure(figsize=(pred_dict["fig_x"], pred_dict["fig_y"]))
        
        if sensors.size > 1:
            sensors = np.sort(sensors)

            for sensor in sensors:
                x = predict_df[predict_df["Sensor"] ==
                            sensor]["CrowdednessCount"].values.reshape(-1, 1)
                hour = predict_df["Hour"].unique().reshape(-1, 1)

                plt.plot(hour, x, label=sensor)
        
        else:
            x = predict_df["CrowdednessCount"].values
            hour = predict_df["Hour"].unique()

            plt.plot(hour, x, label=sensors)

        if pred_dict["model"] in params_dict["clas_models"]:
            plt.yticks(range(1, 5, 1), ("1", "2", "3", "4"))
            
        plt.legend(prop={"size": 15})
        plt.xticks(range(0, 2400, 100), ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13",
                                         "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"))
        plt.xlabel("Hour")
        plt.ylabel("Crowdedness Count")
        plt.title("Time Series Crowdedness")

        plt.savefig(output_dict["plots"] + "{0}_plot.png".format(pred_dict["model"]))

    return predict_df

def prediction(output_dict, params_dict, pred_dict, pbar, i):
    """
    This function calls all function needed to return predictions crowdedness

    Parameters:
    - output_dict (dict): all paths of output files
    - params_dict (dict): general hyperparameters
    - pred_dict (dict): hyperparameters prediction
    - pbar (Bar): progressbar
    - i (i): iteration progressbar
    """

    #Import needed csv files
    full_df = pd.read_csv(output_dict["full_df"])
    passenger_df = pd.read_csv(output_dict["average_passenger_counts"])

    #Save needed stations
    stations = params_dict["stations"]

    #Check if given parameter are valid
    if pred_dict["add_sensors"] == False and pred_dict["extra_coordinate"] == False:
        print("At least, either the custom coordinates or the sensors have to be added")
        pbar.update(i+4)
    else:
        #If add sensors is true, import all present sensors of the full dataset
        if pred_dict["add_sensors"] == True:
            sensors = full_df["Sensor"].unique()

            #Save custom sensor if true
            if pred_dict["extra_coordinate"] == True:
                sensors = np.append("Custom", sensors)
        else:
            sensors = np.array("Custom")

        #Advanced iteration progressbar
        pbar.update(i+1)

        #Import needed models
        model, lat_scaler, lon_scaler, station_scaler, xgbr_model = im.importModels(
            pred_dict["model"], output_dict)

        #Advanced iteration progressbar
        pbar.update(i+1)

        #Construct DF with generated predictions and needed input data for those predictions
        df = generatePredictions(sensors, model, stations, lat_scaler, lon_scaler, station_scaler, full_df, xgbr_model,
                                 passenger_df, output_dict, pred_dict, params_dict)

        #Advanced iteration progressbar
        pbar.update(i+1)

        #Save prediction data to CSV
        df.to_csv(output_dict["predictions"] +
                  "{0}_Predictions.csv".format(pred_dict["model"]), index=False)

        #Advanced iteration progressbar
        pbar.update(i+1)
