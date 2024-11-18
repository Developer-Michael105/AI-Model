import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("housing.csv")

print("data")


#Explore the data


# Display the first few rows
print(data.head())

# Check for missing values and data types
print(data.info())

# Summary statistics
print(data.describe())



#Visualize the data


numerical_data = data.select_dtypes(include=['float64', 'int64'])


correlation_matrix = numerical_data.corr()

# Visualize the correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()





# Histogram for all numerical features
data.hist(figsize=(12, 10), bins=30)
plt.tight_layout()
plt.show()

# Pairplot to visualize relationships
sns.pairplot(data, diag_kind='kde')
plt.show()


#Process the data

# Check missing values
print(data.isnull().sum())

# Fill missing values or drop rows/columns
data = data.fillna(data.mean())  # Example: fill missing values with column mean


data = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop("median_house_value", axis=1))  


#Split the dataset

from sklearn.model_selection import train_test_split

X = data.drop("median_house_value", axis=1)  
y = data["median_house_value"]              

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Train the model


from sklearn.linear_model import LinearRegression

# Initialize and train
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)


#Evaluate the model

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R²: {r2}")


#Save the model

import joblib


joblib.dump(model, "house_price_model.pkl")

# Load the model
loaded_model = joblib.load("house_price_model.pkl")



