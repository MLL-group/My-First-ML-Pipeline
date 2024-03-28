import pickle

import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    df = pd.read_csv('dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=40, shuffle=True)

    clf = RandomForestClassifier(n_estimators=100)

    clf.fit(X_train, y_train)

    with open('RandomForestClassifier.pickle', 'wb') as handle:
        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('RandomForestClassifier.pickle', 'rb') as handle:
        clf = pickle.load(handle)

    y_pred = clf.predict(X_test)

    print("ACCURACY OF THE MODEL:", metrics.accuracy_score(y_test, y_pred))
