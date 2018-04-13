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

A random forest with 800 trees was trained with **workclass_notnull.csv, occupation_notnull.csv & country_notnull.csv** to predict the missing values for **workclass_null.csv, occupation_null.csv & country_null.csv** with scripts [1](https://github.com/pk00095/income-predictor_with_SVC/blob/master/workclass_randomforest.py), [2](https://github.com/pk00095/income-predictor_with_SVC/blob/master/occupation_randomforest.py) & [3](https://github.com/pk00095/income-predictor_with_SVC/blob/master/country_randomforest.py).
