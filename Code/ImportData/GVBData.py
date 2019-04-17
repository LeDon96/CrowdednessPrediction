#Imports
import json
import pandas as pd
import re

#Import Functions other files
import importFiles as im
import exportFiles as ex

def arrivalData(arr_df, stations):

    #Dict to temp save DF's in
    stations_dict = {}

    #Rename 'AantalReizen' column
    arr_df = arr_df.rename(index=str, columns={"AantalReizen": "NumberOfArrivals", "Datum": "Date",
                                            "UurgroepOmschrijving (van aankomst)": "Hour",
                                            "AankomstLat": "AankomstLon", "AankomstLon": "AankomstLat"})
    
    #Only select station in the stations list
    arr_df = arr_df[arr_df["AankomstHalteNaam"].isin(stations)]


    #for station in stations, construct a custom temp df
    for station in stations:
        temp_df = arr_df[arr_df["AankomstHalteNaam"] == station]

        temp_df = temp_df.rename(index=str, columns={"AankomstHalteCode": station + " Code", "AankomstLat": station + " Lat",
                                                        "AankomstLon": station + " Lon", "NumberOfArrivals": station + " Arrivals"}
                                    ).reset_index()

        temp_df = temp_df.groupby(["Date", "Hour"]).agg({station + " Code": 'first',
                                                            station + " Lat": 'first',
                                                            station + " Lon": 'first',
                                                            station + " Arrivals": 'sum'}).reset_index()

        stations_dict["{0}".format(station)] = temp_df

def main():
    #Path to arrival data
    path_to_arr_data = "../../../../Data_thesis/GVB/Datalab_Reis_Bestemming_Uur_20190402.csv"

    #path to departure data
    path_to_dep_data = "../../../../Data_thesis/GVB/Datalab_Reis_Herkomst_Uur_20190403.csv"

    #Stations to be used
    stations = ["Nieuwmarkt", "Nieuwezijds Kolk", "Dam", "Spui"]

    arr_df = im.importCSV(path_to_arr_data, ";")
    dep_df = im.importCSV(path_to_dep_data, ";")

    arrivalData(arr_df, stations)


if __name__ == "__main__":
    main()
