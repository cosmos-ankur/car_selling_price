import pandas as pd 
import numpy as np 
 
dataset = pd.read_csv('C:/Users/Ankur/Desktop/python/ml projects/car price webapp/file.csv')
#print(dataset.head(2))
#print(dataset.columns)
x = dataset[['Present_Price', 'Kms_Driven', 'Owner',
       'year_diff', 'Fuel_Type_Diesel', 'Fuel_Type_Petrol',
       'Seller_Type_Individual', 'Transmission_Manual']]

y = dataset['Selling_Price']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

from sklearn.ensemble import GradientBoostingRegressor
boost = GradientBoostingRegressor()
boost.fit(x_train,y_train)
print(boost.score(x_test,y_test))

import pickle
# open a file, where you ant to store the data
file = open('model.pkl', 'wb')

# dump information to that file
pickle.dump(boost, file)