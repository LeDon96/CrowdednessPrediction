import pickle 

def importModels(model):

    xgbr_model = False

    if model == "rfg":
        model = pickle.load(
            open("../../../Data_thesis/Models/rfg_model.sav", 'rb'))
    elif model == "xgbr":
        model = pickle.load(
            open("../../../Data_thesis/Models/xgbr_model.sav", 'rb'))
        xgbr_model = True
    elif model == "rfc":
        model = pickle.load(
            open("../../../Data_thesis/Models/rfc_model.sav", 'rb'))
    elif model == "xgbc":
        model = pickle.load(
            open("../../../Data_thesis/Models/xgbc_model.sav", 'rb'))

    lat_scaler = pickle.load(
        open("../../../Data_thesis/Models/lat_scaler.sav", 'rb'))
    lon_scaler = pickle.load(
        open("../../../Data_thesis/Models/lon_scaler.sav", 'rb'))
    station_scaler = pickle.load(
        open("../../../Data_thesis/Models/station_scaler.sav", 'rb'))

    return model, lat_scaler, lon_scaler, station_scaler, xgbr_model
