import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # How many features are in the dataset? What is the target column?
    # Dataset metadata
    print(f"10 first records\n{df.head(10)}")

    print(f"metadata\n{df.info()}")
    # Dataset shape
    print(f"shape\n{df.shape}")

    print(f"number of features\n{len(df.columns)}")
    # Alternative
    print(f"number of features\n{df.shape[1]}")

    # Alternative
    print(f"number of rows\n{len(df.index)}")
    # Alternative
    print(f"number of rows\n{df.shape[0]}")

    # What are their types?
    print(f"datatypes\n{df.dtypes}")

    # Do they contain null values?
    print(f"{df.isnull().any()}")
    print(f"{df.isna().any()}")
    print(f"{pd.isnull(df).sum()[pd.isnull(df).sum() >= 0]}")
    print(f"{df.isnull().sum()[pd.isnull(df).sum() >= 0]}")

    # Are there any columns with entirely unique values?
    print("..................")
    print(df.apply(lambda col: col.unique()))
    print("..................")
    print(df.columns[df.nunique() == df.count()])
    print("..................")
    print([col for col in df.columns if df[col].dropna().is_unique])

    # Checking for duplicates
    print(f'The data set contains {sum(df.duplicated())} duplicates')

    # Provide some visualizations - given the target column (‘churn’),
    # create visualizations describing the distribution of values (yes / no),
    # and the proportion of the positive values between demographic attributes
    # such as gender, age, marital status etc.
    df['Churn'].hist()
    plt.show()
    df['gender'].value_counts().plot(kind='bar')
    plt.show()
    pd.Series(df.groupby(['gender'])['gender'].count().plot(kind="bar"))
    plt.show()
    df['tenure'].value_counts().plot.pie()
    plt.show()
    df['TotalCharges'].value_counts().plot()
    plt.show()
