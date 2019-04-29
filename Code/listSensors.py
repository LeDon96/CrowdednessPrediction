import pandas as pd 

def listSensors(df, path):
    df = df.drop_duplicates(subset="Sensor").reset_index()
    df = df[["Sensor", "SensorLatitude", "SensorLongitude"]]

    df.to_csv(path, index=False)