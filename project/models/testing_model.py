import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


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
    X = df.drop(['Result', 'Stakes'], axis=1)
    y = df['Result']
    label_encoder = LabelEncoder()
    X['Race_Type'] = label_encoder.fit_transform(X['Race_Type'])
    X['Going'] = label_encoder.fit_transform(X['Going'])
    y = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)  # 80:10:10 train:validation:test split
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=13)
    return (X_train, y_train, X_validation, y_validation, X_test, y_test)


def load_best_model(sets):
    model = joblib.load('project/models/xgboost/model.joblib')
    model.fit(sets[0], sets[1])
    return model

if __name__ == '__main__':
    sets = split_dataset()
    model = load_best_model(sets)
    test = pd.DataFrame({
        'Odds': 3,
        'Available': 400,
        'Num_Runners': 14,
        'Race_Type': 1,
        'Going': 1,
        'Days': 20, 
        'Bet_Strength': 4
        }, index=[0])
    result = model.predict(test)
    probability = model.predict_proba(test)

    print(result)
    print(probability)