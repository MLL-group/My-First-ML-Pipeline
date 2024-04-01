import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df = pd.read_csv('dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')

    print(df.info())

    # Drop customerID column
    df = df.drop('customerID', axis=1)

    # Drop TotalCharges column: otherwise together with MonthlyCharges you can
    # deduce how many months you have been subscribed
    df = df.drop('TotalCharges', axis=1)

    # The column of Churn is an object data type, so we transform it to a numerical data type.
    df['Churn'] = df['Churn'].map({'No': False, 'Yes': True})

    # creating one-hot columns
    categories = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                  'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'Contract', 'PaperlessBilling', 'PaymentMethod']
    df = pd.get_dummies(df, columns=categories, drop_first=True, dtype=float)

    print(df.info())

    # creating the time and event columns
    event_col = 'Churn'
    time_col = 'tenure'

    # extracting the features/covariables
    features = np.setdiff1d(df.columns, [time_col, event_col]).tolist()
    print(f'Number of features/covariables: {len(features)}')

    # select dependent variables
    y = df.pop('Churn')

    # select independent variables
    X = df

    # # prove that the variables were selected correctly
    print(X.columns)
    #
    # # prove that the variables were selected correctly
    print(y.name)

    # split the data in training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=40, shuffle=True)
    # creating an RF classifier
    clf = RandomForestClassifier(n_estimators=100)

    # Training the model on the training dataset
    # it function is used to train the model using the training sets as parameters
    clf.fit(X_train, y_train)

    with open('RandomForestClassifier.pickle', 'wb') as handle:
        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('RandomForestClassifier.pickle', 'rb') as handle:
        clf = pickle.load(handle)

    # performing predictions on the test dataset
    y_pred = clf.predict(X_test)

    # metrics are used to find accuracy or error
    print("ACCURACY OF THE MODEL:", metrics.accuracy_score(y_test, y_pred))

    # show feature importance
    fi_rfr1 = clf.feature_importances_
    rfi_rfr1 = np.abs(fi_rfr1) / np.sum(np.abs(fi_rfr1))
    lab = X_train.columns
    df_fi_rfr1 = pd.DataFrame(data={'importance_rfr1': rfi_rfr1}, index=lab)
    df_fi_rfr1 = df_fi_rfr1.sort_values(by='importance_rfr1', ascending=True)[18:]
    df_fi_rfr1.plot.barh(y='importance_rfr1')
    plt.show()

    # Display confusion metrics
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.show()
