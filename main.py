import pandas as pd 
import numpy as np 
import json 
import re 
import os
from tqdm import tqdm

#Import own functions
from Code.ImportData.constructFullDataset import constructDF
from Code.Models.models import models
from Code.Prediction.Prediction import prediction

def setUp():
    """
    This function imports all needed hyperparameters and sets up the directories for output data
    """
    #Dict with directories needed input data
    with open("ParamSettings/InputFilePaths.txt", "r") as f:
        input_dict = eval(f.read())

    #Dict with directories for output data
    with open("ParamSettings/OutputFilePaths.txt", "r") as f:
        output_dict = eval(f.read())

    #Dict with general hyperparamters
    with open("ParamSettings/HParams.txt", "r") as f:
        params_dict = eval(f.read())

    #Dict with model specific hyperparameters
    with open("ParamSettings/ModelParams.txt", "r") as f:
        models_dict = eval(f.read())
    
    #Dict with prediction specific hyperparameters
    with open("ParamSettings/PredParams.txt", "r") as f:
        pred_dict = eval(f.read())

    #Construct directories for output data
    if os.path.isdir("Output") == False:
        os.makedirs("Output/Dataset")
        os.makedirs("Output/Results")
        os.makedirs("Output/Models")
        os.makedirs("Output/Visualizations")

    return input_dict, output_dict, params_dict, models_dict, pred_dict

def main():
    """
    This is the main function, that calls on all needed function to generate a crowdedness prediction
    """

    #Instantiate Progress bar
    with tqdm(total=4, desc="Setting up") as pbar:

        #Import all needed hyperparameter data
        input_dict, output_dict, params_dict, models_dict, pred_dict = setUp()

        #Update progress bar
        pbar.update(1)
        pbar.set_description(desc="Importing full dataset")

        #If dataset needs to be constructed
        if params_dict["combine_data"]:

            #Construct full dataset
            constructDF(input_dict, output_dict, params_dict)
        
        #Update progess bar
        pbar.update(1)
        pbar.set_description(desc="Constructing models")

        #If models need to be constructed
        if params_dict["construct_models"]:

            #Construct models
            models(output_dict, params_dict, models_dict, pred_dict)

        #Update progress bar
        pbar.update(1)
        pbar.set_description(desc="Generating prediction")

        #Generate Predictions
        prediction(output_dict, params_dict, pred_dict)

        #Update progress bar
        pbar.update(1)
        pbar.set_description(desc="Finished")
    
    #Close progres bar
    pbar.close()

if __name__ == '__main__':
    main()