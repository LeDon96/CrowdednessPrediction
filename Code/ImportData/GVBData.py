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

    # #Rename 'AantalReizen' column
    # arr_df = arr_df.rename(index=str, columns={"AantalReizen": "NumberOfArrivals", "Datum": "Date",
    #                                         "UurgroepOmschrijving (van aankomst)": "Hour",
    #                                         "AankomstLat": "AankomstLon", "AankomstLon": "AankomstLat"})
    
    #Only select station in the stations list
    arr_df = arr_df[arr_df["AankomstHalteNaam"].isin(stations)]


    #for station in stations, construct a custom temp df
    for station in stations:
        temp_arr_df = arr_df[arr_df["AankomstHalteNaam"] == station]
        temp_dep_df = dep_df[dep_df["VertrekHalteNaam"] == station]

        temp_arr_df = temp_arr_df.rename(index=str, columns={"AankomstHalteCode": station + " Code", "AankomstLat": station + " Lon",
                                                        "AankomstLon": station + " Lat", "AantalReizen": station + " Arrivals",
                                                        "UurgroepOmschrijving (van aankomst)": "Hour", "Datum": "Date"}
                                    )

        temp_dep_df = temp_dep_df.rename(
            index=str, columns={"AantalReizen": station + " Departures", "UurgroepOmschrijving (van vertrek)": "Hour", "Datum": "Date"})

        temp_arr_df = temp_arr_df.groupby(["Date", "Hour"]).agg({station + " Code": 'first',
                                                            station + " Lat": 'first',
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

    return arr_dict[stations[-1]], dep_dict[stations[-1]]



def main():
    #Path to arrival data
    path_to_arr_data = "../../../../Data_thesis/GVB/Datalab_Reis_Bestemming_Uur_20190402.csv"

    #path to departure data
    path_to_dep_data = "../../../../Data_thesis/GVB/Datalab_Reis_Herkomst_Uur_20190403.csv"

    #Stations to be used
    stations = ["Nieuwmarkt", "Nieuwezijds Kolk", "Dam", "Spui"]

    arr_df = im.importCSV(path_to_arr_data, ";")
    dep_df = im.importCSV(path_to_dep_data, ";")

    arr_df, dep_df = stationData(arr_df, dep_df, stations)

    print(arr_df),
    print(dep_df)


if __name__ == "__main__":
    main()
