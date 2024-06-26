# 🤯 Alzheimer's Disease Predictions 🧠

## Table of Contents
- [🤯 Alzheimer's Disease Predictions 🧠](#-alzheimers-disease-predictions-)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Importing Libraries](#importing-libraries)
  - [Data](#data)
  - [Descriptive Analysis](#descriptive-analysis)
    - [Correlation Heatmap](#correlation-heatmap)
    - [Diagnosis Distribution](#diagnosis-distribution)
    - [Boxplot of Features](#boxplot-of-features)
  - [Preprocessing Data](#preprocessing-data)
  - [Machine Learning Models](#machine-learning-models)
    - [Logistic Regression](#logistic-regression)
        - [Confusion Matrix](#confusion-matrix)
      - [Metrics](#metrics)
    - [Random Forest](#random-forest)
        - [Confusion Matrix](#confusion-matrix-1)
      - [Metrics](#metrics-1)
    - [XGBoost](#xgboost)
        - [Confusion Matrix](#confusion-matrix-2)
      - [Metrics](#metrics-2)
  - [Deep Learning](#deep-learning)
    - [Artificial Neural Network (ANN)](#artificial-neural-network-ann)
        - [Accuracy and Loss Plots](#accuracy-and-loss-plots)
        - [Confusion Matrix](#confusion-matrix-3)
      - [Metrics](#metrics-3)
  - [Results](#results)
  - [Conclusion](#conclusion)
  - [References](#references)

## Introduction
This project focuses on predicting Alzheimer's Disease using various machine learning and deep learning models. The dataset used in this project is obtained from `alzheimers_disease_data.csv`.

## Importing Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Keras
- XGBoost

## Data
The dataset is loaded from a CSV file named `alzheimers_disease_data.csv` from [kaggle](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset/data).

## Descriptive Analysis
- Checked dimensions, missing values, data types, and basic statistics of the dataset.
- Dropped the `DoctorInCharge` column as it was not useful for the analysis.

### Correlation Heatmap
![Correlation Heatmap](/correlation.png)

### Diagnosis Distribution
![Diagnosis Distribution](/countplot.png)

### Boxplot of Features
![Boxplot of Features](/boxplot.png)

## Preprocessing Data
- Split the data into features and target variables.
- Split the data into training and testing sets.
- Standardized the data.

## Machine Learning Models

### Logistic Regression
- Performed logistic regression with cross-validation.
- Evaluated the model using accuracy, precision, recall, F1 score, and ROC AUC score.

##### Confusion Matrix
![Logistic Regression Confusion Matrix](/Reg%20Log/confusion_matrix.png)

#### Metrics
Test accuracy (Logistic Regression): `0.8255813953488372`
Test precision (Logistic Regression): `0.7785714285714286`
Test recall (Logistic Regression): `0.7124183006535948`
Test F1 score (Logistic Regression): `0.7440273037542662`
Test ROC AUC (Logistic Regression): `0.8002524716264364`

### Random Forest
- Performed Random Forest classification with cross-validation.
- Evaluated the model using accuracy, precision, recall, F1 score, and ROC AUC score.

##### Confusion Matrix
![Random Forest Confusion Matrix](/Random%20Forest/confusion_matrix.png)

#### Metrics
Test accuracy (Random Forest): `0.9279069767441861`
Test precision (Random Forest): `0.9621212121212122`
Test recall (Random Forest): `0.8300653594771242`
Test F1 score (Random Forest): `0.8912280701754386`
Test ROC AUC (Random Forest): `0.9060074089804394`

### XGBoost
- Performed XGBoost classification with cross-validation.
- Evaluated the model using accuracy, precision, recall, F1 score, and ROC AUC score.

##### Confusion Matrix
![XGBoost Confusion Matrix](XGBoost/confusion_matrix.png)

#### Metrics
Test accuracy (XGBoost): `0.9534883720930233`
Test precision (XGBoost): `0.9235668789808917`
Test recall (XGBoost): `0.9477124183006536`
Test F1 score (XGBoost): `0.9354838709677419`
Test ROC AUC (XGBoost): `0.9521955593308322`

## Deep Learning

### Artificial Neural Network (ANN)
- Defined and trained an ANN model.
- Evaluated the model using accuracy, precision, recall, F1 score, and ROC AUC score.
- Plotted training accuracy and loss.

##### Accuracy and Loss Plots
![ANN Accuracy](ANN/accuracy.png)
![ANN Loss](ANN/loss.png)

##### Confusion Matrix
![ANN Confusion Matrix](ANN/confusion_matrix.png)

#### Metrics

Test accuracy (ANN): `0.827906976744186`
Test precision (ANN): `0.7762237762237763`
Test recall (ANN): `0.7254901960784313`
Test F1 score (ANN): `0.75`
Test ROC AUC (ANN): `0.8049833651872301`

## Results
The project compares the performance of different machine learning and deep learning models for Alzheimer's Disease prediction. The evaluation metrics include accuracy, precision, recall, F1 score, and ROC AUC score.
The best score is with the XGBoost model 

## Conclusion
Based on the results, each model provides different levels of accuracy and performance. The selection of the best model depends on the specific use case and requirements.

## References
- Dataset: `alzheimers_disease_data.csv`
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, keras, xgboost
