# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 19:30:42 2020

@author: ANKUR
"""
import pandas as pd

train_data = pd.read_csv("train.csv")
print(train_data.head())

test_data = pd.read_csv("test.csv")
test_data.head()

from sklearn.ensemble import RandomForestRegressor

y = train_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestRegressor(random_state = 1)
model.fit(X, y)
prediction = model.predict(X_test)
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': prediction})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
