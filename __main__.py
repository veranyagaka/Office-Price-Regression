import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

epochs = 10
learning_rate = 0.0001

data = pd.read_csv('Nairobi Office Price Ex.csv')
target = data['PRICE']
X = data['SIZE']

office_size = np.array(X)
office_price = np.array(target)

print(office_size)
def mse(y_test, y_pred):
    return np.mean((y_test - y_pred) ** 2)


def gradient_descent(x, y, m, c, learning_rate):
    n = len(x)
    y_pred = m * x + c
    dm = (-2 / n) * np.sum(x * (y - y_pred)) 
    dc = (-2 / n) * np.sum(y - y_pred)       
    m = m - learning_rate * dm
    c = c - learning_rate * dc
    return m, c

m, c = np.random.rand(), np.random.rand() 

for epoch in range(epochs):
    m, c = gradient_descent(office_size, office_price, m, c, learning_rate)
    y_pred = m * office_size + c
    error = mse(office_price, y_pred)
    print(f'Epoch {epoch + 1}, MSE: {error}')
plt.scatter(office_size, office_price, color='blue', label='Actual Data')
plt.plot(office_size, y_pred, color='red', label='Best Fit Line')
plt.xlabel('Office Size (sq. ft)')
plt.ylabel('Office Price')
plt.legend()
plt.title('Line of Best Fit after 10 Epochs')
plt.show()

size_to_predict = 100
predicted_price = m * size_to_predict + c
print(f'Predicted office price for {size_to_predict} sq. ft: {predicted_price}')
