#Imports 
import pandas as pd

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


def ImportCrowdednessData(crowd_df, locations_dict, gaww_02, gaww_03):
    """
    This function takes the crowdedness data from all the sensors within Amsterdam. The data from sensors that roughly measure the same place is aggregated under the 
    same sensor name and combined with the latitude and longitude of the sensor's measure area (see function SensorCoordinates). 

    Input:
        - crowd_df: Df with the crowdedness data of all the sensors
        - locations_dict; Dict with the longitude and latitude of the relevant sensor (constructed in function SensorCoordinates)
        - gaww_02/gaww_03: List with alternative sensor names 
    """
    #Group the counts of people per hour, per date, per camera
    crowd_df = crowd_df.groupby(["richting", "datum", "uur"])[
        "SampleCount"].sum().reset_index()
    
    #Rename the columns
    crowd_df = crowd_df.rename(index=str, columns={"richting": "Sensor", "datum": "Date", "uur": "Hour",
                                               "SampleCount": "CrowdednessCount"})
    
    #Insert columns for the sensor coordinates
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

            #Change the ccordinates of the given camera to the correct ones
            v["SensorLongitude"] = locations_dict["GAWW-02"]["Longitude"]
            v["SensorLatitude"] = locations_dict["GAWW-02"]["Latitude"]

        #Change camera name
        elif v["Sensor"] in gaww_03:
            v["Sensor"] = "GAWW-03"

            #Change the ccordinates of the given camera to the correct ones
            v["SensorLongitude"] = locations_dict["GAWW-03"]["Longitude"]
            v["SensorLatitude"] = locations_dict["GAWW-03"]["Latitude"]

        #Mulitply hour with 100 (Same structure as the other files)
        v["Hour"] *= 100

    #Return from Dict
    full_df = pd.DataFrame.from_dict(crowd_dict, orient="index")

    #Onlt save the sensors for which the coordinates are known
    crowd_df = crowd_df[(crowd_df["Sensor"] == "GAWW-02") | (crowd_df["Sensor"] == "GAWW-03")]

    #Group the multiple different sensor data from same date and hour together
    full_df = full_df.groupby(["Sensor", "Date", "Hour", "SensorLongitude",
                                 "SensorLatitude"])["CrowdednessCount"].sum().reset_index()

    return full_df

def main():

    """
    Local Variables
    """
    #Path to crowdedness Data
    path_to_crowdednessData = '../../../../Data_thesis/CMSA/cmsa_data.xlsx'

    #Path to Sensor Data
    path_to_sensorData = '../../../../Data_thesis/Open_Data/crowdedness_sensoren.csv'

    #Sensors to use in Sensor Data
    needed_sensors = ["GAWW-02", "GAWW-03"]

    #Different sensors within a similar area or same sensor with different names
    gaww_02 =  [2, "02R", "2R", "Oude Kennissteeg Occ wifi"] #In the GAWW-02 area
    gaww_03 = [3, "03R"] #In the GAWW-03 area

    #Path to save the file
    csv_path = '../../../../Data_thesis/Full_Datasets/Crowdedness.csv'

    """
    Call on functions
    """

    #Import CSV file
    crowd_df = im.importExcel(path_to_crowdednessData)
    sensor_df = im.importCSV(path_to_sensorData, ";")

    #Transform Sensor df
    locations_dict = SensorCoordinates(sensor_df, needed_sensors)

    #Transform Crowdedness df
    full_df = ImportCrowdednessData(
        crowd_df, locations_dict, gaww_02, gaww_03)

    #Convert DF to CSV
    ex.exportAsCSV(full_df, csv_path)


if __name__ == "__main__":
    main()
