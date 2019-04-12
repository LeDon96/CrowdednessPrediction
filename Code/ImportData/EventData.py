#Imports
import pandas as pd
import datetime

#Import modules other files
import importFiles as im 
import exportFiles as ex

def transformData(events):
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

    #Local variables

    #Dict where all the needed data from each instance will be saved
    events_dict = {}

    #Key for each instance in dict
    key = 0


    #Loop over all events
    for event in events:

        #Save all the dates of each event in a list
        dates = []

        #Date is saved in two different formats in the file

        #Format one --> {'startdate': 'dd-mm-yyyy', 'enddate': 'dd-mm-yyyy'}
        if "startdate" in event["dates"]:

            #Append the events to the list
            dates.append(event["dates"]["startdate"])
            dates.append(event["dates"]["enddate"])

        #Format two --> {'singles': ['dd-mm-yyyy',..., 'dd-mm-yyyy']}
        elif "singles" in event["dates"]:

            #Save entire list to dates 
            dates = event["dates"]["singles"]

        #Loop over each date in dates
        for date in dates:

            #Change type from 'str' to 'datetime'
            date = datetime.datetime.strptime(date, "%d-%m-%Y")
            date = date.date()

            #Set the latitude and longitude of each date of the event to a float
            lat = float(event["location"]["latitude"].replace(",", "."))
            lon = float(event["location"]["longitude"].replace(",", "."))


            #Save all data single instance in temp dict
            event_date = {
                "Date": date, "Event": event["title"], "Latitude": lat, "Longtitude": lon}

            #Append temp to local dict
            events_dict[key] = event_date
            key += 1

    #Convert Dict object to DataFrame and return it
    return pd.DataFrame.from_dict(events_dict, orient="index")

def main():
    """
    This is the main functions that calls on the following functions

    - importFile: Imports Events Database
    - transformData: Transforms imported event data to desired format
    - saveToFile: Saves the output from previous function to CSV file
    """

    #Local Variables

    #path to Event database in JSON format
    json_events_path = "../../../../Data_thesis/Open_Data/Evenementen.json"
    
    #path to desired location to save output events
    csv_path = "../../../../Data_thesis/Full_Datasets/Events.csv"


    #Import JSON file with Event data
    events = im.importFile(json_events_path)

    #Transform data to desired format
    event_df = transformData(events)

    #Convert DF to CSV
    ex.exportAsCSV(event_df, csv_path)


if __name__ == "__main__":
    main()
