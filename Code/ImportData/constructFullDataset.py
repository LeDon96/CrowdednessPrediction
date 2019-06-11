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

def constructDF(input_dict, output_dict, params_dict):
    """
    This function constructs the full needed DF. 

    Parameters:
    - input_dict: dict with all paths to files with needed input data
    - output_dict (dict): all paths of where output files should be saved
    - params_dict (dict): all general hyperparameters that can be changed by user

    Returns: CSV files with all needed data, saved at specified output dir
    """

    #Constructs the sensor datast
    sensor_df = sd.sensorDF(input_dict["coordinateData"], input_dict["blipData"],params_dict["needed_sensors"],
                            input_dict["sensorData"], params_dict["gaww_02"], params_dict["gaww_03"])

    #Constructs the GVB dataset and dataset with all average passenger counts
    gvb_df = gvb.gvbDF(
        input_dict["arrData"], input_dict["deppData"], params_dict["stations"])

    #Constructs the event dataset
    event_df = evd.eventDF(input_dict["eventData"], params_dict["lon_min"], params_dict["lon_max"], 
                           params_dict["lat_min"], params_dict["lat_max"])
    
    #Combines previous constructed datasets
    full_df = cmd.fullDF(sensor_df, gvb_df, event_df,
                         params_dict["stations"], output_dict["lat_scaler"], output_dict["lon_scaler"])

    #Saves DF as CSV
    full_df.to_csv(output_dict["full_df"], index=False)