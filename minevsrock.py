import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

sonar_data = pd.read_csv('Copy of sonar data.csv',header=None)

X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)

model = LogisticRegression()
model.fit(X_train,Y_train)

pickle.dump(model,open('model.pkl','wb'))