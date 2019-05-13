#Imports 
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


def sensorCoordinates(coor_df, needed_sensors):
    """
    This function retrieves the Longitude and Latitude of the needed Sensors and returns these. 

    Parameters:
    - coor_df: DF with longitude and latitude of all the sensors in Amsterdam
    - needed_sensor: List with all the sensors from which the location must be retrieved

    Returns: Dict[SensorName] : {"Longitude": <longitude>, "Latitude": <latitude>}
    """

    #Variables

    #Dict to saved the needed locations in
    locations_dict = {}

    #Select columns to use
    coor_df = coor_df[["Objectnummer", "LNG", "LAT"]]

    #################################################################################

    #Change Df into dict (Needed for operations on data)
    coor_dict = coor_df.to_dict("index")

    for k, v in coor_dict.items():

        #Save only the cameras with the object nummer given above
        if v["Objectnummer"] in needed_sensors:

            #Replace the "." with "," to make sure the coordinates can be turned into floats
            v["LNG"] = float(v["LNG"].replace(",", "."))
            v["LAT"] = float(v["LAT"].replace(",", "."))

            #Save all contents in seperate dict
            locations_dict[v["Objectnummer"]] = {
                "Longitude": v["LNG"], "Latitude": v["LAT"]}

    return locations_dict


def sensorData(sensor_df, blip_df, locations_dict, needed_sensors, gaww_02, gaww_03, lon_scaler_filename, lat_scaler_filename):
    """
    This function takes all the relevant sensor date and combines this in a single DF 

    Parameters:
    - sensor_df (df): Custom made dataframe with subset sensor data
    - blip_df (df): Constructed df from imported blip data
    - locations_dict (dict): contains the longitude and latitude of the relevant sensors
    - needed_sensors (list): selection of given relevant sensors
    - gaww-02 (list): alternate names for the gaww-02 sensor
    - gaww-03 (list): alternate names for the gaww-03 sensor
    - lon_scaler_filename: where the longitude scaler should be saved
    - lat_scaler_filename: where the latitude scaler should be saved

    Returns: DF with all relevant sensor data
    """

    #Variables

    #Scaler to scale the latitude and longitude of sensors
    scaler = StandardScaler()

    #################################################################################

    #Group the counts of people per hour, per date, per camera
    sensor_df = sensor_df.groupby(["richting", "datum", "uur"])[
        "SampleCount"].sum().reset_index()

    #Rename the columns
    sensor_df = sensor_df.rename(index=str, columns={"richting": "Sensor", "datum": "Date", "uur": "Hour",
                                                   "SampleCount": "CrowdednessCount"})

    #Concatenate the two sensor DF's
    sensor_df = pd.concat([sensor_df, blip_df],
                         sort=True).reset_index().drop(columns={"index"})

    #For the longitude number of the sensor
    sensor_df.insert(3, "SensorLongitude", 0)

    #For the latitude number of the sensor
    sensor_df.insert(4, "SensorLatitude", 0)

    #For the number of the day of the week
    sensor_df.insert(3, "weekday", 99)

    #################################################################################

    #Change Df into dict
    crowd_dict = sensor_df.to_dict("index")

    #Loop over dict
    for k, v in crowd_dict.items():

        #Change camera name
        if v["Sensor"] in gaww_02:
            v["Sensor"] = "GAWW-02"

        #Change camera name
        elif v["Sensor"] in gaww_03:
            v["Sensor"] = "GAWW-03"

        #Make the longitude and latitude consistent
        if v["Sensor"] in needed_sensors:

            v["SensorLongitude"] = locations_dict[v["Sensor"]]["Longitude"]
            v["SensorLatitude"] = locations_dict[v["Sensor"]]["Latitude"]

        #Mulitply hour with 100 (Same structure as the other files)
        v["Hour"] *= 100

        #If the hour is 0, transform it to 2400
        if v["Hour"] == 0:
            v["Hour"] = 2400

        #Save the number of the day of the week       
        try:
            v["weekday"] = v["Date"].weekday()
        except:
            #If the above code fails, the date is not timestamp object yet
            v["Date"] = pd.Timestamp.strptime(v["Date"], "%Y-%m-%d")
            v["weekday"] = v["Date"].weekday()

    #Return from Dict
    full_df = pd.DataFrame.from_dict(crowd_dict, orient="index")

    #################################################################################

    #Only save the sensors that are relevant
    full_df = full_df[full_df["Sensor"].isin(needed_sensors)]

    #Group the multiple different sensor data from same date and hour together
    full_df = full_df.groupby(["Sensor", "Date", "Hour", "SensorLongitude",
                               "SensorLatitude", "weekday"])["CrowdednessCount"].sum().reset_index()

    #################################################################################

    #Scale the Longitude and latitude and save the scaler for later use
    full_df["LonScaled"] = scaler.fit_transform(
        full_df["SensorLongitude"].to_numpy().reshape(-1, 1))
    pickle.dump(scaler, open(lon_scaler_filename, 'wb'))

    full_df["LatScaled"] = scaler.fit_transform(
        full_df["SensorLatitude"].to_numpy().reshape(-1, 1))
    pickle.dump(scaler, open(lat_scaler_filename, 'wb'))

    return full_df


def crowdednessDF(path_to_sensorData, path_to_coordinateData, path_to_blipData, needed_sensors, gaww_02, gaww_03, lon_scaler_filename, lat_scaler_filename):

    """
    Call on functions to construct full sensor df

    Parameters:
    - path_to_sensorData (str): path to sensor data custom made
    - path_to_coordinateData (str): path to coordinates data of sensors
    - path_to_blipData (str): path to file with sensor data imported from blip
    - needed_sensors (list): selection of given relevant sensors
    - gaww-02 (list): alternate names for the gaww-02 sensor
    - gaww-03 (list): alternate names for the gaww-03 sensor
    - lon_scaler_filename: where the longitude scaler should be saved
    - lat_scaler_filename: where the latitude scaler should be saved

    Returns: DF with all relevant Sensor data
    """

    #Import CSV file
    sensor_df = pd.read_excel(path_to_sensorData)
    coor_df = pd.read_csv(path_to_coordinateData, sep=";")
    blip_df = pd.read_csv(path_to_blipData)

    #Transform Sensor df
    locations_dict = sensorCoordinates(coor_df, needed_sensors)

    #Transform Crowdedness df
    full_df = sensorData(sensor_df, blip_df, locations_dict,
                         needed_sensors, gaww_02, gaww_03, lon_scaler_filename, lat_scaler_filename)

    return full_df
