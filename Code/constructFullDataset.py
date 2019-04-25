#Imports
import json
import pandas as pd
import re
import numpy as np

#Import Functions other files
import ImportData.importFiles as im
import ImportData.exportFiles as ex
import ImportData.CombineData as cmd 
import ImportData.EventData as evd 
import ImportData.GVBData as gvb 
import ImportData.CrowdednessData as cwd 

# Crowdedness Variables
#Path to original crowdedness dataset
path_to_crowdednessData = '../../../Data_thesis/CMSA/cmsa_data.xlsx'

#Path to Sensor Data
path_to_sensorData = '../../../Data_thesis/Open_Data/crowdedness_sensoren.csv'

#Path to Blip Data
path_to_blipData = "../../../Data_thesis/CMSA/BlipData.csv"

#Sensors to use in Sensor Data
needed_sensors = ["GAWW-01", "GAWW-02", "GAWW-03", "GAWW-04", "GAWW-05", "GAWW-06", "GAWW-07", "GAWW-08", "GAWW-09",
                  "GAWW-10"]

#Alternative names Sensors
gaww_02 = [2, "02R", "2R", "Oude Kennissteeg Occ wifi"]
gaww_03 = [3, "03R"]


# GVB Variables
#Path to arrival data
path_to_arr_data = "../../../Data_thesis/GVB/Datalab_Reis_Bestemming_Uur_20190402.csv"

#path to departure data
path_to_dep_data = "../../../Data_thesis/GVB/Datalab_Reis_Herkomst_Uur_20190403.csv"

#Stations to be used
stations = ["Nieuwmarkt", "Nieuwezijds Kolk", "Dam", "Spui"]


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

#Where to save df
full_df_path = "../../../Data_thesis/Full_Datasets/Full.csv"


def main():

    crowd_df = cwd.CrowdednessDF(path_to_crowdednessData, path_to_sensorData,
                                 path_to_blipData, needed_sensors, gaww_02, gaww_03)

    gvb_df = gvb.GVBDF(path_to_arr_data, path_to_dep_data, stations)

    event_df = evd.EventDF(json_events_path, lon_low,
                           lon_high, lat_low, lat_high, start_date, end_date)
    
    full_df = cmd.CombineDF(crowd_df, gvb_df, event_df)

    #Converts DF to CSV
    full_df.to_csv(full_df_path, index=False)

if __name__ == '__main__':
	main()