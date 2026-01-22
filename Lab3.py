# LAB: Training an ANN with Stochastic Gradient Descent (SGD)
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


#  Loading dataset
data = pd.read_csv("Churn_Modelling.csv")

# Selecting features (X) and output label (y)
X = data.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
y = data["Exited"].values.reshape(-1, 1)


#  Encoding categorical variables (Gender, Geography)
le_gender = LabelEncoder()
le_geo = LabelEncoder()

X["Gender"] = le_gender.fit_transform(X["Gender"])
X["Geography"] = le_geo.fit_transform(X["Geography"])

# Convert to NumPy for ANN
X = X.values

#  Scale the features (important for ANN)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ANN Architecture
input_size = X.shape[1]   
hidden_size = 10          
output_size = 1           
learning_rate = 0.01

epochs = 20    
           
#  Random Weight Initialization
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

#  Activation of Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

#  Train ANN using SGD
for epoch in range(epochs):

    for i in range(X.shape[0]):  
        x_sample = X[i].reshape(1, -1)
        y_true = y[i].reshape(1, -1)

        # ---- Forward Propagation ----
        z1 = np.dot(x_sample, W1) + b1
        a1 = sigmoid(z1)

        z2 = np.dot(a1, W2) + b2
        y_pred = sigmoid(z2)

        # ---- Error ----
        error = y_true - y_pred

        # ---- Backpropagation ----
        d_y_pred = error * sigmoid_derivative(y_pred)

        dW2 = np.dot(a1.T, d_y_pred)
        db2 = d_y_pred

        d_hidden = np.dot(d_y_pred, W2.T) * sigmoid_derivative(a1)

        dW1 = np.dot(x_sample.T, d_hidden)
        db1 = d_hidden

        # ---- Update Weights (SGD) ----
        W1 += learning_rate * dW1
        b1 += learning_rate * db1
        W2 += learning_rate * dW2
        b2 += learning_rate * db2

    print(f"Epoch {epoch+1}/{epochs} completed")

#  Final Predictions
final_output = sigmoid(np.dot(sigmoid(np.dot(X, W1) + b1), W2) + b2)
predictions = (final_output > 0.5).astype(int)

#  Accuracy
accuracy = np.mean(predictions == y)
print("\nFinal Accuracy:", accuracy)
