import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 

# step 1: input (X) and output (y) Data preparation
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# step 2: Initializing and training the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# step 3: Making a prediction for a new value(6)
new_number = np.array([[6]])
prediction = model.predict(new_number)
print(prediction)

# step 4: Visualizing the training data and AI learn pattern
plt.scatter(X, y, color = 'blue')
plt.plot(X, model.predict(X), color = 'red')
plt.title("AI learning the Pattern (y = 2x)")
plt.show()
