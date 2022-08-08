import pandas as pd
import numpy as np
import sklearn


df = pd.read_csv('deploy_df.csv')

df.drop('Unnamed: 0', axis = 1, inplace = True)

x = df.drop('Price', axis = 1)
#print(x.head(2))

y = df['Price']

#splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=10)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier( n_estimators = 1500,  min_samples_split = 15, min_samples_leaf = 2, max_features = 'auto', max_depth = 35)
rf.fit(x_train, y_train)

y_pred =  rf.predict(x_test)

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))

import pickle
pickle.dump(rf, open('model.pkl', 'wb'))s
model = pickle.load(open('model.pkl', 'rb'))
