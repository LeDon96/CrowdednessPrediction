#Imports
import json
import pandas as pd
import re
import numpy as np

#Import Functions other files
import Code.ImportData.CombineData as cmd 
import Code.ImportData.EventData as evd 
import Code.ImportData.GVBData as gvb 
import Code.ImportData.SensorData as sd 

def constructDF(input_dict, output_dict, params_dict, pbar, i):
    """
    This function constructs the full needed DF. 

    Parameters:
    - input_dict: dict with all paths to files with needed input data
    - output_dict: dict with all paths of where output files should be saved
    - params_dict: dict with all general hyperparameters that can be changed by user
    - pbar: Progress bar 
    - i: current number of iteration the progress is at

    Returns: CSV files with all needed data, saved at specified output dir
    """

    #Constructs the sensor datast
    sensor_df = sd.sensorDF(input_dict["sensorData"], input_dict["coordinateData"], input_dict["blipData"],
                            params_dict["needed_sensors"], params_dict["gaww_02"], params_dict["gaww_03"],
                            output_dict["lon_scaler"], output_dict["lat_scaler"])

    #Advanced iteration progressbar
    pbar.update(i+1)

    #Constructs the GVB dataset and dataset with all average passenger counts
    gvb_df, average_df = gvb.gvbDF(input_dict["arrData"], input_dict["deppData"], params_dict["stations"])

    #Advanced iteration progressbar
    pbar.update(i+1)

    #Constructs the event dataset
    event_df = evd.eventDF(input_dict["eventData"], params_dict["lon_low"], params_dict["lon_high"], 
                           params_dict["lat_low"], params_dict["lat_high"])

    #Advanced iteration progressbar
    pbar.update(i+1)
    
    #Combines previous constructed datasets
    full_df = cmd.fullDF(sensor_df, gvb_df, event_df,
                         params_dict["stations"], output_dict["station_scaler"])

    #Advanced iteration progressbar
    pbar.update(i+1)

    #Saves DF as CSV
    full_df.to_csv(output_dict["full_df"], index=False)
    average_df.to_csv(output_dict["average_passenger_counts"], index=True)

    #Advanced iteration progressbar
    pbar.update(i+1)