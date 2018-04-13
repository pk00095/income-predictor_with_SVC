import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC
#from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import LabelEncoder,StandardScaler
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

#Read data into pandas frame
df=pd.read_csv('country_corrected_dataset.csv',na_values=[" "])
df.drop(['education'],inplace=True,axis=1)

#fill NaN with most frequent value in column
df = df.apply(lambda x:x.fillna(x.value_counts().index[0])) 

#Convert text to numbers
enc=LabelEncoder()
df['workclass']=enc.fit_transform(df['workclass'])
df['marital-status']=enc.fit_transform(df['marital-status'])
df['occupation']=enc.fit_transform(df['occupation'])
df['relationship']=enc.fit_transform(df['relationship'])
df['race']=enc.fit_transform(df['race'])
df['sex']=enc.fit_transform(df['sex'])
df['native-country']=enc.fit_transform(df['native-country'])
df['income']=enc.fit_transform(df['income'])

#Initialise feature vectors and ground truth labels
y=df['income']
X=df.drop(['income'],axis=1)

#Scale the input features
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)

#bring outliers to inside range
n=X[:,2]
n[n>2.125]=2.125
X[:,2]=n

#initialise Classifier
svc=SVC(verbose=True)

#parameters={'C':np.arange(0.0001,0.01,0.005)}
parameters={'C':[0.006]}
clf = GridSearchCV(svc, parameters,scoring='precision',cv=4)

#Train classifier
print 'Training classifier...'
clf.fit(X,y)

#Calculate and Print results
print 'calculating scores...'
#scores=cross_val_score(clf, X, y, cv=3, scoring='precision',verbose=True)
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']


for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("Precision :: %0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print 'Precision: {}'.format(clf.grid_scores_ )
print 'Recall: {}'.format(cross_val_score(clf, X, y, cv=3, scoring='recall',verbose=True))
#print 'Recall: {}'.format(cross_val_score(clf, X, y, cv=3,verbose=True))
#scores= cross_val_score(clf, X, y, cv=3,verbose=True)
#print 'Accuracy : {} +/-{}'.format(scores.mean(),scores.std()**2)
print 'Saving model as country_updated_adult.pkl'
joblib.dump(clf,'country_updated_adult.pkl')
