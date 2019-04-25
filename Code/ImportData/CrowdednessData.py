#Imports 
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#Import modules other files
import importFiles as im
import exportFiles as ex


def SensorCoordinates(sensor_df, needed_sensors):
    """
    This function retrieves the Longitude and Latitude of the needed Sensors and returns these. 

    Input:
        - Sensor_df: DF with longitude and latitude of all the sensors in Amsterdam
        - needed_sensor (hyperparameter): List with all the sensors from which the location must be retrieved

    Output:
        - Dict file as follows --> SensorName = {"Longitude": <longitude>, "Latitude": <latitude>}
    """

    #Dict to saved the needed locations in
    locations_dict = {}

    #Select columns to use
    sensor_df = sensor_df[["Objectnummer", "LNG", "LAT"]]

    #Change Df into dict (Needed for operations on data)
    sensor_dict = sensor_df.to_dict("index")

    for k, v in sensor_dict.items():

        #Save only the cameras with the object nummer given above
        if v["Objectnummer"] in needed_sensors:

            #Replace the "." with "," to make sure the coordinates can be turned into floats
            v["LNG"] = float(v["LNG"].replace(",", "."))
            v["LAT"] = float(v["LAT"].replace(",", "."))

            #Save all contents in seperate dict
            locations_dict[v["Objectnummer"]] = {
                "Longitude": v["LNG"], "Latitude": v["LAT"]}

    return locations_dict


def CrowdednessData(crowd_df, blip_df, locations_dict, needed_sensors, gaww_02, gaww_03):
    """
    This function takes the crowdedness data from all the sensors within Amsterdam. The data from sensors that roughly measure the same place is aggregated under the 
    same sensor name and combined with the latitude and longitude of the sensor's measure area (see function SensorCoordinates). 

    Input:
        - crowd_df: Df with the crowdedness data of all the sensors
        - locations_dict; Dict with the longitude and latitude of the relevant sensor (constructed in function SensorCoordinates)
        - gaww_02/gaww_03: List with alternative sensor names 
    """

    crowd_df = pd.concat([crowd_df, blip_df],
                         sort=True).reset_index().drop(columns={"index"})

    #Group the counts of people per hour, per date, per camera
    crowd_df = crowd_df.groupby(["richting", "datum", "uur"])[
        "SampleCount"].sum().reset_index()
    
    #Rename the columns
    crowd_df = crowd_df.rename(index=str, columns={"richting": "Sensor", "datum": "Date", "uur": "Hour",
                                               "SampleCount": "CrowdednessCount"})
    
    #For the longitude number of the sensor
    crowd_df.insert(3, "SensorLongitude", 0)

    #For the latitude number of the sensor
    crowd_df.insert(4, "SensorLatitude", 0)

    #Change Df into dict
    crowd_dict = crowd_df.to_dict("index")

    #Loop over dict
    for k, v in crowd_dict.items():

        #Change camera name
        if v["Sensor"] in gaww_02:
            v["Sensor"] = "GAWW-02"

        #Change camera name
        elif v["Sensor"] in gaww_03:
            v["Sensor"] = "GAWW-03"

        if v["Sensor"] in needed_sensors:

            v["SensorLongitude"] = locations_dict[v["Sensor"]]["Longitude"]
            v["SensorLatitude"] = locations_dict[v["Sensor"]]["Latitude"]

        #Mulitply hour with 100 (Same structure as the other files)
        v["Hour"] *= 100

        if v["Hour"] == 0:
            v["Hour"] = 2400

    #Return from Dict
    full_df = pd.DataFrame.from_dict(crowd_dict, orient="index")

    #Onlt save the sensors for which the coordinates are known
    crowd_df = crowd_df[crowd_df["Sensor"].isin(needed_sensors)]

    #Group the multiple different sensor data from same date and hour together
    full_df = full_df.groupby(["Sensor", "Date", "Hour", "SensorLongitude",
                                 "SensorLatitude"])["CrowdednessCount"].sum().reset_index()

    crowd_df['SensorLongitude'] = LabelEncoder().fit_transform(crowd_df['SensorLongitude'])
    crowd_df['SensorLatitude'] = LabelEncoder().fit_transform(crowd_df['SensorLatitude'])

    return full_df

def main():

    """
    Local Variables
    """
    #Path to crowdedness Data
    path_to_crowdednessData = '../../../../Data_thesis/CMSA/cmsa_data.xlsx'

    #Path to Sensor Data
    path_to_sensorData = '../../../../Data_thesis/Open_Data/crowdedness_sensoren.csv'

    #Path to Blip Data
    path_to_blipData = "../../../Data_thesis/CMSA/BlipData.csv"

    #Sensors to use in Sensor Data
    needed_sensors = ["GAWW-01", "GAWW-02", "GAWW-03", "GAWW-04", "GAWW-05", "GAWW-06", "GAWW-07", "GAWW-08", "GAWW-09",
                      "GAWW-10"]

    #Alternative names Sensors
    gaww_02 = [2, "02R", "2R", "Oude Kennissteeg Occ wifi"]
    gaww_03 = [3, "03R"]

    #Path to save the file
    csv_path = '../../../../Data_thesis/Full_Datasets/Crowdedness.csv'

    """
    Call on functions
    """

    #Import CSV file
    crowd_df = im.importExcel(path_to_crowdednessData)
    sensor_df = im.importCSV(path_to_sensorData, ";")
    blip_df = im.importCSV(path_to_blipData)

    #Transform Sensor df
    locations_dict = SensorCoordinates(sensor_df, needed_sensors)

    #Transform Crowdedness df
    full_df = CrowdednessData(crowd_df, blip_df, locations_dict, needed_sensors, gaww_02, gaww_03)

    #Convert DF to CSV
    ex.exportAsCSV(full_df, csv_path)


if __name__ == "__main__":
    main()
