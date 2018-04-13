import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer

column_names = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hrs-per-week","native-country","income"]

adult_train = pd.read_csv("/home/pratik/Downloads/machine learning/practice/datasets/adult.data",header=None,sep=',\s',na_values=["?"])
adult_train.columns=column_names

ind=adult_train.loc[adult_train[["workclass"]].isnull().any(1)].index
workclass_null= adult_train.iloc[ind,:]
workclass_notnull=adult_train.drop(adult_train.index[ind])
print len(workclass_null)+len(workclass_notnull)
workclass_null.to_csv("workclass_null.csv", sep=',')
workclass_notnull.to_csv("workclass_notnull.csv", sep=',')

ind=adult_train.loc[adult_train[["occupation"]].isnull().any(1)].index
occupation_null=adult_train.iloc[ind,:]
occupation_notnull=adult_train.drop(adult_train.index[ind])
print len(occupation_null)+len(occupation_notnull)
occupation_null.to_csv("occupation_null.csv", sep=',')
occupation_notnull.to_csv("occupation_notnull.csv", sep=',')

ind=adult_train.loc[adult_train[["native-country"]].isnull().any(1)].index
country_null=adult_train.iloc[ind,:]
country_notnull=adult_train.drop(adult_train.index[ind])
print len(country_null)+len(country_notnull)
country_null.to_csv("country_null.csv", sep=',')
country_notnull.to_csv("country_notnull.csv", sep=',')


#ind=adult_train[adult_train[['workclass','native-country','native-country']].isnull().sum(1)==3].index
