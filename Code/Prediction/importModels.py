import pickle 

def importModels(model, output_dict):
    """
    This function imports scalers 

    Parameters:
    - model (str): desired model to generate predictions with
    - output_dict (dict): dict with all paths of output files

    Returns:
    - model: Imported model
    - lat_scaler: Imported latitude scaler
    - lon_scaler: Imported longitude scaler
    - station_scaler: Imported station scaler
    - xgbr_model (boolean): check whether model == xgbr
    """

    #Check whether xgbr model will be used
    xgbr_model = False

    #Import needed model
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

    #Import scaler for sensor Latitudes
    lat_scaler = pickle.load(
        open(output_dict["lat_scaler"], 'rb'))
    
    #Import scaler for sensor Longitudes
    lon_scaler = pickle.load(
        open(output_dict["lon_scaler"], 'rb'))

    #Import scaler for station data
    station_scaler = pickle.load(
        open(output_dict["station_scaler"], 'rb'))

    return model, lat_scaler, lon_scaler, station_scaler, xgbr_model
