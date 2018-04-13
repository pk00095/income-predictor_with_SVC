# income-predictor_with_SVC

## Introduction

This is an attempt to train a classifier to predict an income class of a person and categorise him into either earning **>50k** or **<50k**. We used the publicly available [adult-income](https://archive.ics.uci.edu/ml/datasets/adult) dataset from UCI-dataset repository.

## Preprocessing
First, the data was analysed to find missing values and outliers. Upon finding that **NULL** values were present only in three columns i.e
1. workclass
2. occupation
3. native-country

[This](https://github.com/pk00095/income-predictor_with_SVC/blob/master/NaNseparator.py) script was run on the dataset to separate the rows that contained null values from those that didn't conatin any null values. Six csv files were generated as follows :

* workclass_null.csv
* workclass_notnull.csv
* occupation_null.csv
* occupation_notnull.csv
* country_null.csv
* country_notnull.csv

A random forest with 800 trees was trained with **workclass_notnull.csv, occupation_notnull.csv & country_notnull.csv** to predict the missing values for **workclass_null.csv, occupation_null.csv & country_null.csv** with scripts [1](https://github.com/pk00095/income-predictor_with_SVC/blob/master/workclass_randomforest.py), [2](https://github.com/pk00095/income-predictor_with_SVC/blob/master/occupation_randomforest.py) & [3](https://github.com/pk00095/income-predictor_with_SVC/blob/master/country_randomforest.py). The trained classifiers were stored as **pickle** files which were used by script [4](https://github.com/pk00095/income-predictor_with_SVC/blob/master/replace_nullcountry.py) to create a new csv file named **country_corrected_dataset.csv** where the earlier missing values for country were replaced by those predicted by forests.


# Training
Script [5](https://github.com/pk00095/income-predictor_with_SVC/blob/master/adult_svc.py) was run to train the Support vector machine Classifier upon the preprocessed data and save it as **country_updated_adult.pkl**, during training the **cross validation** was set to 3. It also displays the PRECISION and RECALL score of the classifier on the dataset.

# To-do
Increase the accuracy of the classifier beyond 85%. Add a test script to see performance on new unseen data
