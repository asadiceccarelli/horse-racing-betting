import logging
import json
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO)


def split_dataset():
    """Encodes categoric to numeric in design matrix then splits
    the dataset into train, validation and testing sets in a ratio
    of 80:10:10.
    Returns:
        (tuple): (X_train, y_train, X_validation, y_validation,
            X_test, y_test).
    """
    df = pd.read_csv('project/bet_table.csv')
    df.drop(df[df.Result == 'NON_RUNNER'].index, inplace=True)
    X = df.drop('Result', axis=1)
    y = df['Result']
    label_encoder = LabelEncoder()
    X['Race_Type'] = label_encoder.fit_transform(X['Race_Type'])
    X['Going'] = label_encoder.fit_transform(X['Going'])
    y = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)  # 80:10:10 train:validation:test split
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=13)
    return (X_train, y_train, X_validation, y_validation, X_test, y_test)


def get_classification_report(model, sets):
    """Calculates the precision, recall and F1 score of each set in
    a classification report.
    Args:
        model (class): The classification model used in predictions.
        sets (tuple): (X_train, y_train, X_validation, y_validation,
            X_test, y_test).
    Returns:
        reports_dict (dict): A dictionary of each sets classification report.
    """
    y_train_pred = model.predict(sets[0])
    y_validation_pred = model.predict(sets[2])
    y_test_pred = model.predict(sets[4])
    train_report = classification_report(sets[1], y_train_pred, output_dict=True)
    validation_report = classification_report(sets[3], y_validation_pred, output_dict=True)
    test_report = classification_report(sets[5], y_test_pred, output_dict=True)
    reports_dict = {'train_report': train_report, 'validation_report': validation_report, 'test_report': test_report}
    return reports_dict


def save_model_metrics(model_name, model, best_params, reports_dict):
    """Saves the metrics and model as .json and .joblib files
    respectively.
    Args:
        model_name (str): The model directory to save information in.
        model (class): The classification model used in predictions.
        best_params (dict): The best hyperparameters of the tuned model.
        reports_dict (dict): A dictionary of the classification report.
    """
    joblib.dump(model, open(f'project/models/{model_name}/model.joblib', 'wb'))
    json.dump(reports_dict, open(f'project/models/{model_name}/reports.json', 'w'))
    json.dump(best_params, open(f'project/models/{model_name}/parameters.json', 'w'))


def create_baseline_model(sets):
    """Creates a baseline model using Logistic Regression and saves
    the model and metrics.
    """
    model = LogisticRegression(solver='newton-cg')
    model.fit(sets[0], sets[1])
    params = model.get_params()
    reports_dict = get_classification_report(model, sets)
    save_model_metrics('logistic_regression', model, params, reports_dict)


def tune_random_forest(sets):
    """Creates and tunes the hyperparameters of a Random Forest
    Classifier and saves the model and metrics.
    """
    parameters = dict(
            n_estimators=list(range(80, 90)),
            max_depth=list(range(2, 7)),
            max_samples = list(range(30, 40))
    )
    model = RandomForestClassifier(random_state=13)
    kfold = KFold(n_splits=5, shuffle=True, random_state=13)
    grid_search = GridSearchCV(model, parameters, cv=kfold)
    grid_search.fit(sets[0], sets[1])
    best_params = grid_search.best_params_
    model = RandomForestClassifier(**best_params, random_state=13)
    model.fit(sets[0], sets[1])
    reports_dict = get_classification_report(model, sets)
    save_model_metrics('random_forest', model, best_params, reports_dict)


def tune_xgboost(sets):
    """Creates and tunes the hyperparameters of a XGB Classifier
    and saves the model and metrics.
    """
    parameters = dict(
            n_estimators=list(range(1, 20)),
            max_depth=list(range(2, 7)),
            min_child_weight=list(range(10, 16)),
            learning_rate=np.arange(0.1, 1, 0.1),
    )
    model = xgb.XGBClassifier(random_state=13)
    kfold = KFold(n_splits=5, shuffle=True, random_state=13)
    grid_search = GridSearchCV(model, parameters, cv=kfold)
    grid_search.fit(sets[0], sets[1])
    best_params = grid_search.best_params_
    model = xgb.XGBClassifier(**best_params, random_state=13)
    model.fit(sets[0], sets[1])
    reports_dict = get_classification_report(model, sets)
    save_model_metrics('xgboost', model, best_params, reports_dict)


if __name__ == '__main__':
    sets = split_dataset()
    # create_baseline_model(sets)
    # tune_random_forest(sets)
    tune_xgboost(sets)