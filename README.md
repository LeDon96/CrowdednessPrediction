# Prediction Models to estimate crowdedness within Amsterdam

## Abstract
Amsterdam is a crowded city, which increases the pressure on public services, public events and public spaces. Sensors were placed in the city to measure the crowdedness of pedestrians within small predetermined areas. This research aims to train prediction models that predict the crowdedness at and between predetermined locations within Amsterdam, at an hourly rate for each single day within a predetermined period. 

In order to train the prediction models, historical data from the following sources was used; (i) public transport data from predetermined stations (ii) crowdedness counts from sensors located at the predetermined locations (iii) dates of events within Amsterdam. 

The models used in this research were *Random Forest* and *XGBoost*, for both regression and classification. The regression models predicted the numerical pedestrian counts and the classification models predicted the crowdedness level of the pedestrians, based on the crowdedness distribution in the historical dataset. 

With the predictions made at the sensor locations, the *XGBoost* models outperformed the *Random Forest* models. 

As a method of testing the predictions between the known sensor locations, the models generalized the data of the known sensor locations to the unknown location. To evaluate the model performance, the data of a single sensor was removed during the training of the models and the crowdedness of that sensor was predicted during the evaluation phase. Which was repeated for each given sensor and the average performance results were evaluated. 

With the generalizations to unknown sensor locations, the regression *XGBoost* models also outperformed the regression *Random Forest* models. But, the performance of the classification *XGBoost* and classification *Random Forest* was equal. 

## Documents
- [Thesis](Documents/Thesis%20Crowdedness.pdf)
- [Thesis Proposal](Documents/Thesis_Proposal_Crowdedness.pdf)
- [Code Requirements](Documents/Requirements.md)
- [Hyperparameter Settings Explanation](Documents/Hyperparameters.md)

## Prediction
The predictions can be generated in two different forms:
- **Prediction for unknown dates**: The prediction will be made at a known location for unknown dates
- **Generalized prediction**: The prediction will be made at a unknown location for known dates

## Usage
- Clone or donwload the code
- Set the [Hyperparameter Settings](ParamSettings/)
- Run [main.py](main.py)
    - The dataset only needs to be constructed if not present
    - The models only need to be constructed if not present 
- Enjoy the given prediciton :)

## Code
All the code can be run from the [main.py](main.py) file. This files calls on all the needed functions to do the following:
- [Import the datasets](Code/ImportData/constructFullDataset.py)
- [Construct models](Code/Models/models.py)
- [Generate Predictions](Code/Prediction/Prediction.py)
- [Jupyter Notebooks with some data exploration](Jupyter%20Notebooks)

All the above steps need to be completed on the first run, afterwards, only step 3 needs to be done to generate the predictions. 