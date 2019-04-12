#Imports 
import pandas as pd

#Import modules other files
import importFiles as im
import exportFiles as ex

def CrowdednessDF(df):
    #Group the counts of people per hour, per date, per camera
    df = df.groupby(["richting", "datum", "uur"])[
        "SampleCount"].sum().reset_index()
    
    #Rename the columns
    df = df.rename(index=str, columns={"richting": "Sensor", "datum": "Date", "uur": "Hour",
                                               "SampleCount": "CrowdednessCount"})
    
    #Insert columns for the sensor coordinates
    #For the longitude number of the sensor
    df.insert(3, "SensorLongitude", 0)

    #For the latitude number of the sensor
    df.insert(4, "SensorLatitude", 0)


def main():
    #Locat variables
    path_to_crowdednessdata = '../../../../Data_thesis/CMSA/cmsa_data.xlsx'

    #Import CSV file
    crowd_df = im.importExcel(path_to_crowdednessdata)

    #Transform Crowdedness df
    crowd_df = CrowdednessDF(crowd_df)
