import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import StandardScaler        # Import to scale the data
from sklearn.linear_model import LogisticRegression     # To import logistic regression algorithm
from sklearn.metrics import accuracy_score              # Import to find the accuracy of the model

df = pd.read_csv('/Users/vivan/Codes/placement.csv')

# Pre-processing the data to remove a useless column (no null values since basic dataset)
df = df.iloc[:, 1:]


# EDA to figure out which ml algorithm to use to train model
plt.scatter(df['cgpa'],df['iq'],c=df['placement'])
plt.xlabel("CGPA")
plt.ylabel("IQ")
plt.show()                   
# Since graph can be seen to form 2 distinct groups we use logistic regression and also since this is a classification problem
# Also used for N distinct groups (advanced version or similar algorithm) where the output can be categorised (i.e not numbers)


# Implementing logistic regression 
X = df.iloc[: , 0:2]         # Independent Variable (since cgpa & iq are unrelated)
Y = df.iloc[:, -1]           # Dependent Variable (since placement depends upon iq & cgpa)
X_training ,X_testing ,Y_training, Y_testing = train_test_split(X,Y,test_size=0.1) 
# Test Size 0.1 means 1% of the dataset (always considers the rows only not the columns)


# To Scale the training data (X & Y) now
scaler = StandardScaler()
X_training = scaler.fit_transform(X_training)
X_testing = scaler.transform(X_testing)


# Training the model on the X_training set & Y_training set
clf = LogisticRegression()
clf.fit(X_training,Y_training)
Y_prediction = clf.predict(X_testing)        # Model gives the predicted output for the testing data we set aside during train test split


# Now we evaluate the model's accuracy 
acc = accuracy_score(Y_testing,Y_prediction)      # Checking the 'placement' testing data with the models prediction
print(acc)