from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC

df=pd.read_csv("workclass_notnull.csv")
df.drop(df.columns[0],axis=1,inplace=True) #first column contains index

df.drop(['income'],axis=1,inplace=True)
y=np.array(df['workclass'])
y=y.ravel() #convert column vector to 1-D array
#print len(np.unique(y))
X=df.drop(['workclass','occupation','native-country','education'],axis=1) #education-num is numerical representation of education

enc = LabelEncoder()
X['marital-status']=enc.fit_transform(X['marital-status'])
X['relationship']=enc.fit_transform(X['relationship'])
X['sex']=enc.fit_transform(X['sex'])
X['race']=enc.fit_transform(X['race'])
y=enc.fit_transform(y)

print 'Initialising random forest with 800 trees....'
#forest = RandomForestClassifier(n_estimators=1000,random_state=0,n_jobs=-1,verbose=True,min_samples_leaf=50)
forest=SVC(C=10,verbose=True)
print 'training forest...'
forest.fit(X,y)
print 'training complete ...'
print 'saving...'
joblib.dump(forest,"workclass_svc.pkl")
print 'Saved model as workclass_random_forest.pkl ....'
