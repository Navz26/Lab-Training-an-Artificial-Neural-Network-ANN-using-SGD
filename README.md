Lab – Training an Artificial Neural Network (ANN) using SGD


1)  Objective
This lab demonstrates how to build and train an Artificial Neural Network (ANN) from scratch using Stochastic Gradient Descent (SGD). The goal is to predict customer churn (Exited) based on customer attributes.


2)  Dataset
Churn_Modelling.csv
Contains customer demographic and banking information
Target variable: Exited (1 = churned, 0 = not churned)


3)  Methodology
Preprocessed data by:
Encoding categorical variables (Gender, Geography)
Scaling numerical features using StandardScaler
Implemented a 2-layer ANN:
Input layer → Hidden layer (10 neurons) → Output layer
Used Sigmoid activation for both layers
Trained the network using Stochastic Gradient Descent
Updated weights after each training sample


4)  Model Configuration
Learning Rate: 0.01
Hidden Neurons: 10
Epochs: 20
Loss: Binary classification error
Output Threshold: 0.5


5)  Output
Binary predictions for customer churn
Final model accuracy printed after training


6)  Tools & Libraries
Python
NumPy
pandas
scikit-learn


7)  Course
 CIS 380
Lab: ANN with Stochastic Gradient Descent
