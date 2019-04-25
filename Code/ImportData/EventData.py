#Imports
import pandas as pd
import datetime

#Import modules other files
import importFiles as im 
import exportFiles as ex

def transformData(events, lat_low, lat_high, lon_low, lon_high, start_date, end_date):
    """
    This function transforms the data of each each event in the following dict structure:
    - Date: A single date of the given event
        - type: datetime 
    - Event Name: Name of the given event
        - type: str
    - Latitude: Latitude of the place of the given event
        - type: float
    - Longitude: Longitude of the place of the given event
        - type: float

    Keep the following in mind:
    - Each event has multiple instance --> Each event is saved per date. So an event with 5 dates, will have 5 instances
    - It's possible that a single date has multiple occurrences --> There is no unique ID per row
    - It's possible that multiple different events take place at the same coordinates (longitude, latitude)

    - Input: JSON object with events
    - Output: DataFrame object with all the given events, saved per the above mentioned criteria
    """

    #Dict where all the needed data from each instance will be saved
    events_dict = {}

    #Key for each instance in dict
    key = 0


    #Loop over all events
    for event in events:

        #Save all the dates of each event in a list
        dates = []

        #Set the latitude and longitude of each date of the event to a float
        lat = float(event["location"]["latitude"].replace(",", "."))
        lon = float(event["location"]["longitude"].replace(",", "."))

        #Check if Longitude and Latitude between specified parameters
        if lon > lon_low and lon < lon_high and lat > lat_low and lat < lat_high:

            #Check if saved in format one or two

            #Format one --> {'startdate': 'dd-mm-yyyy', 'enddate': 'dd-mm-yyyy'}
            if "startdate" in event["dates"]:

                #Append the events to the list
                dates.append(event["dates"]["startdate"])
                dates.append(event["dates"]["enddate"])

            #Format two --> {'singles': ['dd-mm-yyyy',..., 'dd-mm-yyyy']}
            elif "singles" in event["dates"]:

                #Save entire list to dates
                dates = event["dates"]["singles"]

            #Change type from 'str' to 'datetime'
            dates = [pd.Timestamp.strptime(date, "%d-%m-%Y") for date in dates]

            for date in dates:
                if start_date < date < end_date:

                    #Dict with all data single event
                    event_date = {"Date": date, "is_event": 1.0}

                    #Append dict to list
                    events_dict[key] = event_date
                    key += 1

    #Convert Dict object to DataFrame and return it
    return pd.DataFrame.from_dict(events_dict, orient="index")


def EventDF(json_events_path, lon_low, lon_high, lat_low, lat_high, start_date, end_date):
    """
    This is the main functions that calls on the following functions

    - importFile: Imports Events Database
    - transformData: Transforms imported event data to desired format
    - saveToFile: Saves the output from previous function to CSV file
    """
    #Import JSON file with Event data
    events = im.importJSON(json_events_path)

    #Transform data to desired format
    event_df = transformData(events, lat_low, lat_high, lon_low, lon_high, start_date, end_date)

    return event_df