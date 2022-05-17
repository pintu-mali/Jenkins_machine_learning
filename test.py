import joblib
model = joblib.load("model.pk1")
print(model.predict([[1]]))
