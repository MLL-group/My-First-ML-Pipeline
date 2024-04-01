import dataclasses
import json
import os
from dataclasses import dataclass
from datetime import datetime

import git
import pandas as pd
from nyoka import skl_to_pmml
from pypmml import Model
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


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


def train_model(X_train, y_train, config):
    clf = RandomForestClassifier(n_estimators=config.n_estimators, max_depth=config.max_depth)
    clf.fit(X_train, y_train)
    return clf


def save_model(model, features, target, config):
    path = f"{config.name}.pmml"
    pipeline = Pipeline([
        ("model", model)
    ])
    skl_to_pmml(pipeline, features, target, path)

    return os.stat(path)


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


def predict(model, X_test):
    y_pred = model.predict(X_test)
    y_pred = y_pred.iloc[:, :1]
    y_pred = y_pred['predicted_Churn'].map({'False': False, 'True': True})

    return y_pred


def load_model(config):
    path = f"{config.name}.pmml"
    model = Model.fromFile(path)
    return model


def split_dataset(config):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size,
                                                        random_state=40, shuffle=True)
    return X_train, X_test, y_train, y_test


def save_metadata(experiment_context, experiment):
    with open(f"{experiment.name}.metadata.json", 'w') as f:
        json.dump(experiment_context, f)
        # print(experiment_context, file=f)


if __name__ == "__main__":
    experiment = Experiment(
        name="Telco-Customer-Churn-Classification-RF",
        dataset=Dataset(test_size=0.3, path='dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv'),
        hyperparameters=Hyperparameters(n_estimators=100, max_depth=None)
    )

    experiment_context = {
        'experiment': dataclasses.asdict(experiment),
        'user': os.environ.get("USER"),
        'create_time': datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        'git_commit': git.Repo(search_parent_directories=True).head.object.hexsha,
        'git_branch': git.Repo(search_parent_directories=True).active_branch.name,
    }

    current_time = datetime.now()
    X, y = prepare_dataset(experiment.dataset)
    experiment_context['data_preparation_time'] = datetime.now().timestamp() - current_time.timestamp()

    experiment_context['X'] = json.loads(X.describe().to_json())
    experiment_context['y'] = json.loads(y.describe().to_json())

    X_train, X_test, y_train, y_test = split_dataset(experiment.dataset)

    current_time = datetime.now()
    trained_model = train_model(X_train, y_train, experiment.hyperparameters)
    experiment_context['trained_model_time'] = datetime.now().timestamp() - current_time.timestamp()

    current_time = datetime.now()
    file_stats = save_model(model=trained_model, features=X_train.columns, target="Churn", config=experiment)
    experiment_context['save_model_time'] = datetime.now().timestamp() - current_time.timestamp()

    experiment_context['model_file_size_in_MB'] = file_stats.st_size / (1024 * 1024)

    current_time = datetime.now()
    loaded_model = load_model(config=experiment)
    experiment_context['loaded_model_time'] = datetime.now().timestamp() - current_time.timestamp()

    current_time = datetime.now()
    y_pred = predict(loaded_model, X_test)
    experiment_context['inference_time'] = datetime.now().timestamp() - current_time.timestamp()

    experiment_context['accuracy_score'] = metrics.accuracy_score(y_test, y_pred)

    save_metadata(experiment_context, experiment)
