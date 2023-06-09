import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the car dataset (CSV file)
car_data = pd.read_csv('car_data.csv')

# Split the dataset into features (X) and target variable (y)
X = car_data.drop('Price', axis=1)
y = car_data['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Example: Predicting car price for new data
new_data = pd.DataFrame({
    'Mileage': [50000],
    'Year': [2018],
    'Brand': ['Toyota']
})

price_pred = model.predict(new_data)
print("Predicted Car Price:", price_pred)
