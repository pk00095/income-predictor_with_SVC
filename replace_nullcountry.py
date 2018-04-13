import pandas as pd


df1=pd.read_csv("country_null.csv")
df2=pd.read_csv("country_predictions.csv")

#print df1.head()
#print df2.head()
df2.set_index('index')
#print df2.ix[14]
df1['native-country']=df2['predictions']

#print df1.head()
df1.drop(df1.columns[0],inplace=True,axis=1)
df1.to_csv('country_predicted_dataset.csv',sep=',',index=False)

df3=pd.read_csv("country_notnull.csv")
df3.drop(df3.columns[0],inplace=True,axis=1)

#print df3.head()

df4=pd.concat([df3,df1],ignore_index=True)
df4 = df4.sample(frac=1).reset_index(drop=True)
df4.to_csv('country_corrected_dataset.csv',sep=',',index=False)
