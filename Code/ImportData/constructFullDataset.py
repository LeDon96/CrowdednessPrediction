#Imports
import json
import pandas as pd
import re
import numpy as np

#Import Functions other files
import ImportData.CombineData as cmd 
import ImportData.EventData as evd 
import ImportData.GVBData as gvb 
import ImportData.SensorData as sd 

# Sensor Variables
#Path to original crowdedness dataset
path_to_sensorData = '../../../Data_thesis/CMSA/cmsa_data.xlsx'

#Path to Sensor Data
path_to_coordinateData = '../../../Data_thesis/Open_Data/crowdedness_sensoren.csv'

#Path to Blip Data
path_to_blipData = "../../../Data_thesis/CMSA/BlipData.csv"

#Where to save Longitude Scaler
lon_scaler_filename = "../../../Data_thesis/Models/lon_scaler.sav"

#Where to save Latitude Scaler
lat_scaler_filename = "../../../Data_thesis/Models/lat_scaler.sav"

#Sensors to use in Sensor Data
needed_sensors = ["GAWW-01", "GAWW-02", "GAWW-03","GAWW-04", "GAWW-05", "GAWW-06", "GAWW-07"]

#Alternative names Sensors
gaww_02 = [2, "02R", "2R", "Oude Kennissteeg Occ wifi"]
gaww_03 = [3, "03R"]

#################################################################################

# GVB Variables
#Path to arrival data
path_to_arr_data = "../../../Data_thesis/GVB/Datalab_Reis_Bestemming_Uur_20190402.csv"

#path to departure data
path_to_dep_data = "../../../Data_thesis/GVB/Datalab_Reis_Herkomst_Uur_20190403.csv"

#Stations to be used
stations = ["Nieuwmarkt", "Nieuwezijds Kolk", "Dam", "Spui", "Centraal Station"]

#################################################################################

#Event Variables
#path to Event database in JSON format
json_events_path = "../../../Data_thesis/Open_Data/Evenementen.json"

#Parameters for area to search in
#longitude
lon_low = 4.88
lon_high = 4.92

#Latitude
lat_low = 52.36
lat_high = 52.39

#Start date for relevant events
start_date = pd.Timestamp(2018, 3, 11)

#End date for relevant events
end_date = pd.Timestamp(2019, 4, 30)

#################################################################################

#Combine Variables

#Where to save the station weight scaler
station_scaler_filename = "../../../Data_thesis/Models/station_scaler.sav"

#################################################################################

#Where to save df
full_df_path = "../../../Data_thesis/Full_Datasets/Full.csv"
average_df_path = "../../../Data_thesis/Full_Datasets/AveragePassengerCounts.csv"


def main():

    sensor_df = sd.sensorDF(path_to_sensorData, path_to_coordinateData, path_to_blipData,
                            needed_sensors, gaww_02, gaww_03, lon_scaler_filename, lat_scaler_filename)

    gvb_df, average_df = gvb.gvbDF(
        path_to_arr_data, path_to_dep_data, stations)

    event_df = evd.eventDF(json_events_path, lon_low,
                           lon_high, lat_low, lat_high, start_date, end_date)
    
    full_df = cmd.fullDF(sensor_df, gvb_df, event_df, stations, station_scaler_filename)

    #Converts DF to CSV
    full_df.to_csv(full_df_path, index=False)
    average_df.to_csv(average_df_path, index=True)

if __name__ == '__main__':
	main()
