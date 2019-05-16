import pickle 

def importModels(model, output_dict):

    xgbr_model = False

    if model == "rfg":
        model = pickle.load(
            open(output_dict["rfg_model"], 'rb'))
    elif model == "xgbr":
        model = pickle.load(
            open(output_dict["xgbr_model"], 'rb'))
        xgbr_model = True
    elif model == "rfc":
        model = pickle.load(
            open(output_dict["rfc_model"], 'rb'))
    elif model == "xgbc":
        model = pickle.load(
            open(output_dict["xgbc_model"], 'rb'))

    lat_scaler = pickle.load(
        open(output_dict["lat_scaler"], 'rb'))
    lon_scaler = pickle.load(
        open(output_dict["lon_scaler"], 'rb'))
    station_scaler = pickle.load(
        open(output_dict["station_scaler"], 'rb'))

    return model, lat_scaler, lon_scaler, station_scaler, xgbr_model
