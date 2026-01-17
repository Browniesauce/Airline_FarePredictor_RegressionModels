import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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
reg = linear_model.LinearRegression()
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