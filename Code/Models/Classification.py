from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from yellowbrick.classifier import ClassPredictionError
import pickle
import matplotlib.pyplot as plt

def hyperParameter(x_train, y_train, score, model, model_name, cycles, params):
    """
    This function fits multiple hyperparameters to find the optimal combinationf.

    Parameters:
    - x_train (df): training feature set model
    - y_train (df): training target set model
    - score (str): scoring metric used to find the best model 
    - model (model): model that needs to be evaluated
    - model_name (str): unique ID model
    - cycles (int): number of iterations to test the model
    - params (dict): optimal model hyperparameters

    Returns:
    - Dict with the optimal combination of parameters
    - Score of the best model
    """

    #Call on the the hyper parameter fitting modle
    hyp = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=cycles, scoring=score, n_jobs=4, cv=10,
                             refit=score, random_state=42)

    #Run hyper parameter fitting
    ## XGB Classifier model only takes values as input
    if model_name == "xgbr":
        model = hyp.fit(x_train.drop(
            columns={"Date"}).values, y_train["CrowdednessCount"].values)
    else:
        model = hyp.fit(x_train.drop(
            columns={"Date"}), y_train["CrowdednessCount"])

    return model.best_params_, model.best_score_

def trainModel(model, x_train, y_train, kf, train_dates, labels, params, model_name, kf_size):
    """
    This function trains the classification model with optimal parameters

    Parameters:
    - model (model): model that needs to be trained
    - x_train (df): training features set model
    - y_train (df): training target set model
    - train_dates (list): all dates in the training set
    - labels (list): all class labels
    - params (dict): optimal model hyperparameters
    - model_name (str): unique ID model
    - kf_size (int): number of splits in KFolds

    Returns:
    - mean_acc: Mean Accuracy
    - prec_dict: Mean Precision per class
    - rec_dict: Mean Recall per class
    - f1_dict: Mean F1-Score per class
    - model: Trained model
    """

    #If model is baseline, don't include random state/n_jobs parameter
    if model_name == "dc":
        model.set_params(**params)
    else:
        model.set_params(**params)
        params["random_state"] = 42
        params["n_jobs"] = 4

    #Variables
    mean_acc = 0
    mean_precision = 0
    mean_recall = 0
    mean_f1_score = 0

    prec_dict = {}
    rec_dict = {}
    f1_dict = {}

    #Split the train dates into train and test, kf_size times
    for train_index, test_index in kf.split(train_dates):

        #Select rows dataset based on the dates in train_index
        ## XGB Classifier model only takes values as input
        if model_name == "xgbr":
            x = x_train[x_train["Date"].isin(
                train_dates[train_index])].drop(columns={"Date"}.values)
            y = y_train[y_train["Date"].isin(
                train_dates[train_index])]["CrowdednessCount"].values 
        else: 
            x = x_train[x_train["Date"].isin(
                train_dates[train_index])].drop(columns={"Date"})
            y = y_train[y_train["Date"].isin(
                train_dates[train_index])]["CrowdednessCount"]

        #Fit the model
        model.fit(x, y)

        #Select rows dataset based on dates in test_index 
        ## XGB Classifier model only takes values as input
        if model_name == "xgbr":
            x_test = x_train[x_train["Date"].isin(
                train_dates[test_index])].drop(columns={"Date"}).values
            y_test = y_train[y_train["Date"].isin(
                train_dates[test_index])]["CrowdednessCount"].values
        else: 
            x_test = x_train[x_train["Date"].isin(
                train_dates[test_index])].drop(columns={"Date"})
            y_test = y_train[y_train["Date"].isin(
                train_dates[test_index])]["CrowdednessCount"]

        #Make predictions
        y_pred_model = model.predict(x_test)

        #If model is baseline, only calculate accraucy
        if model_name == "dc":
            mean_acc += accuracy_score(y_test, y_pred_model)
        else:
            #Calculate precision, recall and f1 score per label and accuracy of entire set
            mean_acc += accuracy_score(y_test, y_pred_model)
            mean_precision += precision_score(y_test, y_pred_model, average=None)
            mean_recall += recall_score(y_test, y_pred_model, average=None)
            mean_f1_score += f1_score(y_test, y_pred_model, average=None)

    #Take the mean score over all 10 iterations
    mean_acc = round(((mean_acc / kf_size) * 100), 2)
    mean_precision = (mean_precision / kf_size) * 100
    mean_recall = (mean_recall / kf_size) * 100
    mean_f1_score = (mean_f1_score / kf_size) * 100

    #Return the precision, recall, and f1score in seperate dicts
    if model_name != "dc":
        for i in range(len(labels)):
            prec_dict["{0}".format(labels[i])] = mean_precision[i]
            rec_dict["{0}".format(labels[i])] = mean_recall[i]
            f1_dict["{0}".format(labels[i])] = mean_f1_score[i]

    return mean_acc, prec_dict, rec_dict, f1_dict, model

def evalModel(model, x_eval, y_eval, labels, visualization, plot_dir, model_name, x_train, y_train):
    """
    This function evaluates the trained model on unseen data

    Parameters:
    - model (model): model that needs to be evaluated
    - x_eval (df): test features model
    - y_eval (df): test target model
    - labels (list): class labels
    - visualization (bool): whether you want a scatter model of evaluation results
    - plot_dir (str): directory where plots have to be saved
    - model_name (str): name of the model
    - x_train (df): training feature set model
    - y_train (df): training target set model

    Returns:
    - acc: Accuracy
    - prec_dict: Precision per class
    - rec_dict: Recall per class
    - f1_dict: F1-Score per class
    """

    #Dicts where the results will be saved
    prec_dict = {}
    rec_dict = {}
    f1_dict = {}

    # XGB Classifier model only takes values as input
    if model_name == "xgbr":
        x_eval = x_eval.values
        y_eval = y_eval.values

    #Predict target values
    y_pred_eval = model.predict(x_eval)

    #If model is baseline, only calculate accruacy
    if model_name == "dc":
        acc = accuracy_score(y_eval, y_pred_eval)
    else:
        #Calculate precision, recall and f1 score per label and accuracy
        acc = accuracy_score(y_eval, y_pred_eval)
        prec = precision_score(y_eval, y_pred_eval, average=None)
        rec = recall_score(y_eval, y_pred_eval, average=None)
        f1 = f1_score(y_eval, y_pred_eval, average=None)

        #Return the precision, recall, and f1score in seperate dicts
        for i in range(len(labels)):
            prec_dict["{0}".format(labels[i])] = prec[i]
            rec_dict["{0}".format(labels[i])] = rec[i]
            f1_dict["{0}".format(labels[i])] = f1[i]

    #Visualize the model results
    if visualization == True:
        visualizer = ClassPredictionError(
                model
            )
        visualizer.fit(x_train.drop(columns={"Date"}), y_train["CrowdednessCount"])
        visualizer.score(x_eval, y_eval)
        visualizer.poof("{0}{1}.png".format(plot_dir, model_name))
        plt.gcf().clear()

    return acc, prec_dict, rec_dict, f1_dict


def modelConstruction(model_dir, plot_dir, model_name, model, labels, x_train, y_train, x_eval, y_eval, score, train_dates, kf, cycles, visualization, params, kf_size):
    """
    This function trains a linear regression model

    Parameters:
    - model_dir (str): directory where model has to be saved
    - plot_dir (str): directory where plots have to be saved
    - model_name (str): name of the model
    - model (model): model that needs to be evaluated
    - labels (list): class labels
    - x_train (df): training feature set model
    - y_train (df): training target set model
    - x_eval (df): test features set model
    - y_eval (df): test target set model
    - score (str): sklearn standard scoring metric used by model
    - train_dates (list): dates present in training model 
    - kf (model): used to split training dates into training and test k times
    - cycles (int): number of iterations to test the model
    - visualization (bool): whether you want a scatter model of evaluation results
    - params: dict with optimal model hyperparameters
    - kf_size (int): number of splits in KFolds

    Returns: Dict containing all metrics of hyperparameter, training and evaluation of the model
    """

    #Dict to save all the results in
    results_dict = {}

    #Hyper parameter tuning
    best_params, best_score = hyperParameter(
        x_train, y_train, score, model, model_name, cycles, params)
    results_dict["Hyper R2 Score"] = best_score
    results_dict["Model Parameters"] = best_params

    #Train the model
    train_acc, train_prec, train_rec, train_f1, model = trainModel(
        model, x_train, y_train, kf, train_dates, labels, best_params, model_name, kf_size)

    #Save results training
    results_dict["Train Accuracy Score"] = train_acc
    results_dict["Train Precision Score"] = train_prec
    results_dict["Train Recall Score"] = train_rec
    results_dict["Train F1 Score"] = train_f1

    #Evaluate model on unseen data
    eval_acc, eval_prec, eval_rec, eval_f1 = evalModel(
        model, x_eval, y_eval, labels, visualization, plot_dir, model_name, x_train, y_train)
   
    #Save results evaluation
    results_dict["Evaluation Accuracy Score"] = eval_acc
    results_dict["Evaluation Precision Score"] = eval_prec
    results_dict["Evaluation Recall Score"] = eval_rec
    results_dict["Evaluation F1 Score"] = eval_f1

    #Save model as pickle
    filename = "{0}{1}_model.sav".format(model_dir, model_name)
    pickle.dump(model, open(filename, 'wb'))

    return results_dict
