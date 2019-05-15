from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from yellowbrick.classifier import ClassPredictionError
import pickle
import matplotlib.pyplot as plt

def hyperParameter(x_train, y_train, score, model, cycles, **params):
    """
    This function fits multiple hyperparameters to find the optimal combinationf.

    Parameters:
    - x_train (df): training features model
    - y_train (df): training target model
    - score (str): scoring metric used to find the best model 
    - model (model): model that needs to be evaluated
    - cycles (int): number of iterations to test the model
    - params: dict with optimal model hyperparameters

    Returns:
    - Dict with the optimal combination of parameters
    - Score of the best model
    """

    #Call on the the hyper parameter fitting modle
    hyp = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=cycles, scoring=score, n_jobs=4, cv=10,
                             random_state=42, refit=score)

    #Run hyper parameter fitting
    model_model = hyp.fit(x_train.drop(
        columns={"Date"}), y_train["CrowdednessCount"])

    return model_model.best_params_, model_model.best_score_

def trainModel(model, x_train, y_train, kf, train_dates, labels, params, model_name):

    if model_name == "dc":
        model.set_params(**params)
    else:
        model.set_params(**params)
        params["random_state"] = 42
        params["n_jobs"] = 4

    mean_acc = 0
    mean_precision = 0
    mean_recall = 0
    mean_f1_score = 0

    prec_dict = {}
    rec_dict = {}
    f1_dict = {}

    for train_index, test_index in kf.split(train_dates):

        x = x_train[x_train["Date"].isin(
            train_dates[train_index])].drop(columns={"Date"})
        y = y_train[y_train["Date"].isin(
            train_dates[train_index])]["CrowdednessCount"]

        model.fit(x, y)

        x_test = x_train[x_train["Date"].isin(
            train_dates[test_index])].drop(columns={"Date"})
        y_test = y_train[y_train["Date"].isin(
            train_dates[test_index])]["CrowdednessCount"]

        y_pred_model = model.predict(x_test)

        if model_name == "dc":
            mean_acc += accuracy_score(y_test, y_pred_model)
        else:
            mean_acc += accuracy_score(y_test, y_pred_model)
            mean_precision += precision_score(y_test, y_pred_model, average=None)
            mean_recall += recall_score(y_test, y_pred_model, average=None)
            mean_f1_score += f1_score(y_test, y_pred_model, average=None)
        


    mean_acc = round(((mean_acc / 10) * 100), 2)
    mean_precision = (mean_precision / 10) * 100
    mean_recall = (mean_recall / 10) * 100
    mean_f1_score = (mean_f1_score / 10) * 100

    if model_name != "dc":
        for i in range(len(labels)):
            prec_dict["{0}".format(labels[i])] = mean_precision[i]
            rec_dict["{0}".format(labels[i])] = mean_recall[i]
            f1_dict["{0}".format(labels[i])] = mean_f1_score[i]

    return mean_acc, prec_dict, rec_dict, f1_dict, model

def evalModel(model, x_eval, y_eval, labels, visualization, plot_dir, x_train, y_train, model_name):

    prec_dict = {}
    rec_dict = {}
    f1_dict = {}

    y_pred_eval = model.predict(x_eval)

    if model_name == "dc":
        acc = accuracy_score(y_eval, y_pred_eval)
    else:
        acc = accuracy_score(y_eval, y_pred_eval)
        prec = precision_score(y_eval, y_pred_eval, average=None)
        rec = recall_score(y_eval, y_pred_eval, average=None)
        f1 = f1_score(y_eval, y_pred_eval, average=None)

        for i in range(len(labels)):
            prec_dict["{0}".format(labels[i])] = prec[i]
            rec_dict["{0}".format(labels[i])] = rec[i]
            f1_dict["{0}".format(labels[i])] = f1[i]

    if visualization == True:
        visualizer = ClassPredictionError(
                model
            )


        visualizer.fit(x_train.drop(columns={"Date"}), y_train["CrowdednessCount"])
        visualizer.score(x_eval, y_eval)
        visualizer.finalize()
        plt.savefig("{0}{1}.png".format(plot_dir, model))

    return acc, prec_dict, rec_dict, f1_dict


def modelConstruction(model_dir, plot_dir, model_name, model, labels, x_train, y_train, x_eval, y_eval, score, train_dates, kf, cycles, visualization, **params):
    """
    This function trains a linear regression model

    Parameters:
    - model_dir (str): directory where model has to be saved
    - model_name (str): name of the model
    - model (model): model that needs to be evaluated
    - labels (list): class labels
    - x_train (df): training features model
    - y_train (df): training target model
    - x_eval (df): test features model
    - y_eval (df): test target model
    - score (str): sklearn standard scoring metric used by model
    - train_dates (list): dates present in training model 
    - kf (model): used to split training dates into training and test k times
    - cycles (int): number of iterations to test the model
    - visualization (bool): whether you want a scatter model of evaluation results
    - params: dict with optimal model hyperparameters

    Returns: Dict containing all metrics of hyperparameter, training and evaluation of the model
    """

    results_dict = {}

    best_params, best_score = hyperParameter(
        x_train, y_train, score, model, cycles, **params)
    results_dict["Hyper R2 Score"] = best_score
    results_dict["Model Parameters"] = best_params

    train_acc, train_prec, train_rec, train_f1, model = trainModel(
        model, x_train, y_train, kf, train_dates, labels, best_params, model_name)

    results_dict["Train Accuracy Score"] = train_acc
    results_dict["Train Precision Score"] = train_prec
    results_dict["Train Recall Score"] = train_rec
    results_dict["Train F1 Score"] = train_f1

    eval_acc, eval_prec, eval_rec, eval_f1 = evalModel(
        model, x_eval, y_eval, labels, visualization, plot_dir, x_train, y_train, model_name)
    results_dict["Evaluation Accuracy Score"] = eval_acc
    results_dict["Evaluation Precision Score"] = eval_prec
    results_dict["Evaluation Recall Score"] = eval_rec
    results_dict["Evaluation F1 Score"] = eval_f1

    filename = "{0}{1}_model.sav".format(model_dir, model_name)
    pickle.dump(model, open(filename, 'wb'))

    return results_dict
