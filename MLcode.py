import numpy as np
import pandas as pd
import joblib
data = pd.read_csv("salary_data.csv")
x = data.drop('Salary',axis=1)
y = data[['Salary']]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
joblib.dump(model,"model.pk1")
print("Successfully Trained the model")
