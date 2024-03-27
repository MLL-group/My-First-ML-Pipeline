# Tutorial 
https://docs.google.com/document/d/10DUEniYQQDmbNI_f19s63n_Terjo0TtMMflw5liOuus/edit

# Dataset  
source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn


# Example:
* https://medium.com/analytics-vidhya/churn-prediction-in-a-telco-70ba5aa12f70
* code: https://github.com/alonsosilvaallende/Churn-Prediction-in-a-Telco/blob/master/Churn-Prediction-in-a-Telco_colab.ipynb

# Theory
https://mll-group.github.io/My-First-ML-Pipeline/#1

* https://www.geeksforgeeks.org/decision-tree/
* https://www.geeksforgeeks.org/random-forest-classifier-using-scikit-learn/

## Exercise 1: Data exploration
1. How many features are in the dataset? What is the target column?
2. What are their types?
3. Do they contain null values?
4. Are there any columns with entirely unique values?
5. Provide some visualizations - given the target column (‘churn’), create visualizations describing the distribution of values (yes / no), and the proportion of the positive values between demographic attributes such as gender, age, marital status etc.
6. Display the feature importance

source: [eda.py](./eda.py)

## Exercise 2 : Simple model
1. Split the data randomly, 70/30.
2. Train a random forest model and evaluate the scores. 
3. Try to change the n_estimators parameter to see score improvements. 
4. Save the model (pickel), and note its size and computation time.
5. Display confusion metrics and the model scores.

source: [simple_model.py](./simple_model.py)

---
# TBD

---

## Exercise 3: Clean and prep the dataset
1. the column TotalCharges was wrongly detected as an object. This column represents the total amount charged to the customer and it is, therefore, a numeric variable. For further analysis, we need to transform this column into a numeric data type.
2. Drop columns with the lowest importance score (on exercise 1) and retrain the model in exercise 2 to compare the results
3. Use label encoding for the categorical features - replace the string values with numeric values. Use SKlearn default method. A nice example here -
https://gist.github.com/amandaiglesiasmoreno/2f451cc0a6966e997b936e9a3c49352a/raw/a7c84da048eb6ad4b72a351028657cbe0fdc120c/label_encoding.py
4. Normalize the numeric columns - 'tenure', 'MonthlyCharges', 'TotalCharges'
An example code here - https://gist.github.com/amandaiglesiasmoreno/13502116e5684b853f27296413e0bd68/raw/a847efd015d8e897a402489c096d8da5504b7eb8/min_max_normalization.py

source: [prep_dataset.py](./prep_dataset.py)

## Exercise 4: Modeling
1. Compare several estimators as in the example below:
     https://gist.github.com/amandaiglesiasmoreno/e20a4f052cf48e4a163db3eeb7d1fcd6/raw/e93903223d93af623b8711f5c3409b343458e9b7/create_models.py
     https://gist.github.com/amandaiglesiasmoreno/e0928eae6e1bc3bccad408b9ad00ccd6/raw/73120590f57c631b02ffdfbc5ebbda88437a085d/test_models.py
2. Choose the best performing algorithm, and manually tune the hyperparameters.
3. Use grid search for hyperparameters tuning - which method yields better results?

---


https://github.com/Galdina/FS_mod3_project
https://www.kaggle.com/code/gauravduttakiit/telecom-churn-case-study-with-random-forest
https://medium.com/@zachary.james.angell/applying-survival-analysis-to-customer-churn-40b5a809b05a
