import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Student data
X = np.array([
    [8, 90, 85, 80],
    [7, 85, 80, 75],
    [9, 95, 90, 88],
    [6, 80, 75, 70],
    [5, 75, 65, 60],
    [4, 65, 55, 50],
    [3, 60, 45, 40],
    [2, 55, 40, 35],
    [8, 92, 88, 82],
    [7, 87, 82, 78],
    [9, 96, 92, 90],
    [4, 68, 50, 45]
])

# 1 = Pass, 0 = Fail
y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0])

# Split into training and unseen test students
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Simple neural network
model = MLPClassifier(
    hidden_layer_sizes=(20, 20),
    max_iter=2000,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Performance
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

print("\nActual Test Results :", y_test)
print("Predicted Test Results:", test_pred)

# Check for overfitting
difference = train_accuracy - test_accuracy

if difference > 0.15:
    print("\nThe model shows signs of OVERFITTING.")
else:
    print("\nThe model does not show significant overfitting.")
