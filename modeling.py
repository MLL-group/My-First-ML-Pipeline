from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV


@dataclass
class Hyperparameters:
    n_estimators: int
    max_depth: int | None


@dataclass
class Dataset:
    test_size: float
    path: str


@dataclass
class Experiment:
    hyperparameters: Hyperparameters
    dataset: Dataset
    name: str


def prepare_dataset(config):
    df = pd.read_csv(config.path)
    df['Churn'] = df['Churn'].map({'No': False, 'Yes': True})
    # creating one-hot columns
    categories = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                  'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'Contract', 'PaperlessBilling', 'PaymentMethod']
    df = pd.get_dummies(df, columns=categories, drop_first=True, dtype=float)
    df = df[["tenure",
             "MonthlyCharges",
             "Contract_Two year",
             "PaymentMethod_Electronic check",
             "InternetService_Fiber optic",
             "Churn", ]]
    numeric_columns = ["tenure",
                       "MonthlyCharges",
                       "Contract_Two year",
                       "PaymentMethod_Electronic check",
                       "InternetService_Fiber optic", ]
    # scale numerical variables using min max scaler
    for column in numeric_columns:
        # minimum value of the column
        min_column = df[column].min()
        # maximum value of the column
        max_column = df[column].max()
        # min max scaler
        df[column] = (df[column] - min_column) / (max_column - min_column)

    y = df.pop('Churn')
    X = df

    return X, y


def split_dataset(config):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size,
                                                        random_state=40, shuffle=True)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    experiment = Experiment(
        name="Telco-Customer-Churn-Classification-RF",
        dataset=Dataset(test_size=0.3, path='dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv'),
        hyperparameters=Hyperparameters(n_estimators=100, max_depth=None)
    )

    X, y = prepare_dataset(experiment.dataset)

    X_train, X_test, y_train, y_test = split_dataset(experiment.dataset)

    print("---------------------------------------------------------")

    model = RandomForestClassifier(n_estimators=50, max_depth=None)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(classification_report(y_test, predictions))

    print("---------------------------------------------------------")
    print("Grid search:")

    model = RandomForestClassifier()
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}

    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    predictions = best_model.predict(X_test)

    print(f"best_params {best_params}")
    print(classification_report(y_test, predictions))
