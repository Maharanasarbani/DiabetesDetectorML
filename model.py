import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle
from sklearn.metrics import accuracy_score

dataset=pd.read_csv('data/diabetes_dataset.csv')
X=dataset.drop(columns='Outcome',axis=1)
Y=dataset['Outcome']

scaler=StandardScaler()
scaler.fit(X)
pickle.dump(scaler, open('model/scaler.pkl', 'wb'))
standadized_data=scaler.transform(X)


X=standadized_data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
classifier=svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)
pickle.dump(classifier, open('model/model.pkl', 'wb'))