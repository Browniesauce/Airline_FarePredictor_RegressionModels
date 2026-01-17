import pandas as pd
import numpy as np
import math
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


df = pd.read_csv("Clean_Dataset.csv")
Airlines = df.airline.value_counts()
Start_City = df.source_city.value_counts()
Destination_City = df.destination_city.value_counts()
Departure_Time = df.departure_time.value_counts()
Arrival_Time = df.arrival_time.value_counts()
Stops = df.stops.value_counts()
Ticket_Class = df['class'].value_counts()
Minimum_Time = df['duration'].min()
Maximum_Time = df['duration'].max()
Median_Time = df['duration'].median()

print(df)
print(Airlines)
print(Start_City)
print(Destination_City)
print(Departure_Time)
print(Arrival_Time)

#Pre Processing the Input Data from CSV file
df = df.drop("Unnamed: 0",axis=1)
df = df.drop("flight",axis=1)

df['class'] = df['class'].apply(lambda x:1 if x=="Business" else 0)
df.stops = pd.factorize(df.stops)[0]
df = df.join(pd.get_dummies(df.airline,prefix = 'Airline')).drop('airline',axis=1)
df = df.join(pd.get_dummies(df.source_city,prefix = 'Source')).drop('source_city',axis=1)
df = df.join(pd.get_dummies(df.destination_city,prefix = 'Destination')).drop('destination_city',axis=1)
df = df.join(pd.get_dummies(df.arrival_time,prefix = 'Arrival')).drop('arrival_time',axis=1)
df = df.join(pd.get_dummies(df.departure_time,prefix = 'Departute')).drop('departure_time',axis=1)

#Training the Regression Model 
X , Y = df.drop('price',axis = 1) , df.price
X_Train , X_Test , Y_Train , Y_Test = train_test_split(X , Y , test_size = 0.2)
reg = RandomForestRegressor(n_jobs =- 1)
reg.fit(X_Train , Y_Train)
reg.score(X_Test , Y_Test)

#Evaluation the Model
y_pred = reg.predict(X_Test)

R2_Score = r2_score(Y_Test,y_pred)
Mean_Abs_Err = mean_absolute_error(Y_Test,y_pred)
Mean_Sq_Err = mean_squared_error(Y_Test,y_pred)
Root_Mean_Sq_Err = math.sqrt(Mean_Sq_Err)
print("R2 Score: ",R2_Score , "Mean Absolute Error: ",Mean_Abs_Err , "Mean Squared Error: ", Mean_Sq_Err , "Root Mean Squared Error:" , Root_Mean_Sq_Err)

#Plotting the Prediction VS Actual Price
plt.scatter(Y_Test,y_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predicton VS Actual Flight Pricing')

importances = dict(zip(reg.feature_names_in_, reg.feature_importances_))
sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)

print(sorted_importances)

plt.figure(figsize=(20, 6))
plt.bar([x[0] for x in sorted_importances[:10]] , [x[1] for x in sorted_importances[:10]])

#Hyper Parameter Tuning

reg = RandomForestRegressor(n_jobs = -1)

param_dist ={
     'n_estimators': randint(100,300),
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': randint(2,11),
    'min_samples_leaf': randint(1,5),
    'max_features': [1.0, 'sqrt']
}

random_search = RandomizedSearchCV(reg, param_distributions=param_dist, n_iter=2, cv=3, scoring= 'neg_mean_squared_error', verbose= 2, random_state= 10, n_jobs = -1)
random_search.fit(X_Train, Y_Train)

best_regressor = random_search.best_estimator_
best_regressor.score(X_Test, Y_Test)

y_pred_1 = best_regressor.predict(X_Test)

R2_Score1 = r2_score(Y_Test,y_pred_1)
Mean_Abs_Err1 = mean_absolute_error(Y_Test,y_pred_1)
Mean_Sq_Err1 = mean_squared_error(Y_Test,y_pred_1)
Root_Mean_Sq_Err1 = math.sqrt(Mean_Sq_Err)
print(R2_Score1,Mean_Abs_Err1,Mean_Sq_Err1,Root_Mean_Sq_Err1)

#Plotting Fine Tuned Regressor Model using Hyper Parameter Tuning
plt.scatter(Y_Test,y_pred_1)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predicton VS Actual Flight Pricing')

#Prediction Test Using Test DataSet
Y_Pred_2 = best_regressor.predict(X_Test)
predicted_df = X_Test.copy()
predicted_df['Predicted_Price'] = Y_Pred_2
print(predicted_df.head())

#Exporting Predicted Prices as CSV File
predicted_df.to_csv('predicted_flight_prices.csv', index=False)
print("DataFrame successfully exported to 'predicted_flight_prices.csv'")