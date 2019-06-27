# CrowdednessPrediction

## Abstract
The aim of this research is to construct a model that predicts the crowdedness at selected spots within the city of Amsterdam, in the form of a time series. Historical data will be gathered from localized crowdedness measure points and will be combined with GVB data. The predictions with the given data will be made with both a multivariate regression model and a gradient boosted decision tree. The predicted crowdedness of both models will be visualized in separate time series over a given period of time. The two models will be compared in the evaluation based on the loss. 

## Prediction
The predictions can be generated in two different forms:
- **Prediction for unknown dates**: The prediction will be made at a known location for unknown dates
- **Generalized prediction**: The prediction will be made at a unknown location for known dates

## Usage
- Clone or donwload the code
- Set the [Hyperparameter Settings](ParamSettings/)
- Run [main.py](main.py)
- Enhoy the given prediciton :)

## Documents
- [Thesis Design](Documents/Thesis_Proposal_Crowdedness.pdf)
- [Code Requirements](Documents/Requirements.md)
- [Hyperparameter Settings Explanation](Documents/Hyperparameters.md)

## Code
All the code can be run from the [main.py](main.py) file. This files calls on all the needed functions to do the following:
- [Import the datasets](Code/ImportData/constructFullDataset.py)
- [Construct models](Code/Models/models.py)
- [Generate Predictions](Code/Prediction/Prediction.py)
- [Jupyter Notebooks with some data exploration](Jupyter%20Notebooks)

All the above steps need to be completed on the first run, afterwards, only step 3 needs to be done to generate the predictions. 