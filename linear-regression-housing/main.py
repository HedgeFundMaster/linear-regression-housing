import matplotlib.pyplot as plt

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the California housing dataset 
data = fetch_california_housing(as_frame=True)
df = pd.DataFrame(data.data, columns=data.feature_names)
target = data.target 

# # Use only median income (MedInc) as the predictor
X = df[['MedInc']] 
y = target 

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model 
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Intercept: {model.intercept_:4f}")
print(f"Coefficients for MedInc: {model.coef_[0]:4f}")
print(f"Mean Squared Error: {mse:4f}")
print(f"R^2 Score: {r2:4f}")

plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Linear Regression: MedInc vs. House Value")
plt.legend()
plt.savefig("regression_plot.png")
print("📈 Plot saved as regression_plot.png")
