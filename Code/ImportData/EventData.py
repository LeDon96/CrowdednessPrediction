#Imports
import pandas as pd
import datetime
import json

def transformData(events, lat_low, lat_high, lon_low, lon_high):
    """
    This function transforms all present dates between start and end date in the following, given 
    that the coordinatees of the event fall between the given longitude and latitude borders:
    - Date(datetime): date
    - is_event(float): there is an event on the given date

    Parameters:
    - events (json): dataset events 
    - Coordinate borders:
        - lon_low: min value longitude
        - lon_high: max value longitude
        - lat_low: min value Latitude
        - lat_high: max value Latitude

    Returns:DF with relevant event data
    """

    #Variables

    #Dict where all the needed data from each instance will be saved
    events_dict = {}

    #Key for each instance in dict
    key = 0

    #################################################################################

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

            #Save present date with confirmation that there is an event
            for date in dates:

                #Dict with all data single event
                event_date = {"Date": date, "is_event": 1.0}

                #Append dict to list
                events_dict[key] = event_date
                key += 1

    #Convert Dict object to DataFrame and return it
    return pd.DataFrame.from_dict(events_dict, orient="index")


def eventDF(json_events_path, lon_low, lon_high, lat_low, lat_high):
    """
    This is the main functions that constructs the full events DF, by calling the needed function

    Parameters:
    - json_events_path: path to event dataset
    - Coordinate borders:
        - lon_low: min value longitude
        - lon_high: max value longitude
        - lat_low: min value Latitude
        - lat_high: max value Latitude

    Returns: Event DF
    """

    #Import JSON file with Event data
    with open(json_events_path) as file_data:

        #Save data as JSON object
        events = json.load(file_data)

    #Transform data to desired format
    event_df = transformData(events, lat_low, lat_high, lon_low, lon_high)

    return event_df