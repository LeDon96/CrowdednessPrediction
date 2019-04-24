#Imports
import json
import pandas as pd
import re

#Import Functions other files
import importFiles as im
import exportFiles as ex

def stationData(arr_df, dep_df, stations):

    #Dict to temp save DF's in
    arr_dict = {}
    dep_dict = {}
    
    #Only select station in the stations list
    arr_df = arr_df[arr_df["AankomstHalteNaam"].isin(stations)]


    #for station in stations, construct a custom temp df
    for station in stations:
        temp_arr_df = arr_df[arr_df["AankomstHalteNaam"] == station]
        temp_dep_df = dep_df[dep_df["VertrekHalteNaam"] == station]

        temp_arr_df = temp_arr_df.rename(index=str, columns={"AankomstLat": station + " Lon",
                                                        "AankomstLon": station + " Lat", "AantalReizen": station + " Arrivals",
                                                        "UurgroepOmschrijving (van aankomst)": "Hour", "Datum": "Date"}
                                    )

        temp_dep_df = temp_dep_df.rename(
            index=str, columns={"AantalReizen": station + " Departures", "UurgroepOmschrijving (van vertrek)": "Hour", "Datum": "Date"})

        temp_arr_df = temp_arr_df.groupby(["Date", "Hour"]).agg({station + " Lat": 'first',
                                                            station + " Lon": 'first',
                                                            station + " Arrivals": 'sum'}).reset_index()

        temp_dep_df = temp_dep_df.groupby(["Date", "Hour"]).agg(
            {station + " Departures": 'sum'}).reset_index()

        arr_dict["{0}".format(station)] = temp_arr_df
        dep_dict["{0}".format(station)] = temp_dep_df

    for i in range(len(stations)-1):
        arr_dict[stations[i+1]] = pd.merge(arr_dict[stations[i]],
                                           arr_dict[stations[i+1]], on=["Date", "Hour"], how="outer")

        dep_dict[stations[i+1]] = pd.merge(dep_dict[stations[i]],
                                           dep_dict[stations[i+1]], on=["Date", "Hour"], how="outer")

    return pd.merge(arr_dict[stations[-1]], dep_dict[stations[-1]],
             on=["Date", "Hour"], how="outer")

def TransformData(df):

    #Variables
    date_format_1 = '%d/%m/%Y %H:%M:%S'
    date_format_2 = '%m/%d/%Y %H:%M:%S'
    
    #Fill NaN values with 0
    df = df.fillna(0.0)

    #Add column day numbers
    df["weekday"] = 99

    #Add whether column to indicate whether it is weekend
    df["is_weekend"] = 0

    #Dataframe to Dict
    df_dict = df.to_dict("index")

    #Loop over dict
    for k, v in df_dict.items():
        #Replace time string with time blok
        time_blok = v["Hour"][:5]
        time_blok = re.sub('[:]', '', time_blok)
        v["Hour"] = int(time_blok)

        if v["Hour"] == 0:
            v["Hour"] = 2400

        #Remove AM/PM from string
        v["Date"] = v["Date"][:-3]
        try:
            #Transform the date string to datatime.date object
            date = pd.Timestamp.strptime(v["Date"], date_format_1)
            #Transfrom date to weekday number
            v["weekday"] = date.weekday()
        except:
            #Transform the date string to datatime.date object
            date = pd.Timestamp.strptime(v["Date"], date_format_2)

            #Transfrom date to weekday number
            v["weekday"] = date.weekday()
        
        #Transform Date string to datetime object
        v["Date"] = date.date()

        #Check if weekday is in the weekend
        if date.weekday() == 5 or date.weekday() == 6:
            v["is_weekend"] = 1

        v["Date"] = date.date()

    return pd.DataFrame.from_dict(df_dict, orient="index")



def main():
    #Path to arrival data
    path_to_arr_data = "../../../../Data_thesis/GVB/Datalab_Reis_Bestemming_Uur_20190402.csv"

    #path to departure data
    path_to_dep_data = "../../../../Data_thesis/GVB/Datalab_Reis_Herkomst_Uur_20190403.csv"

    #Stations to be used
    stations = ["Nieuwmarkt", "Nieuwezijds Kolk", "Dam", "Spui"]

    #Path to save the file
    csv_path = '../../../../Data_thesis/Full_Datasets/GVBData.csv'

    arr_df = im.importCSV(path_to_arr_data, ";")
    dep_df = im.importCSV(path_to_dep_data, ";")

    full_df = stationData(arr_df, dep_df, stations)

    full_df = TransformData(full_df)

    ex.exportAsCSV(full_df, csv_path)


if __name__ == "__main__":
    main()
