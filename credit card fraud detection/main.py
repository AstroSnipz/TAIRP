#importing modules
import numpy as np
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#load the dataset
data = pd.read_csv("credit_card_fraud_detection/creditcard.csv")

#first 5 and last five rows of dataset
print(data.head())
print(data.tail())

#dataset information
data.info()

#checking the number of missing values in each column
print(data.isnull().sum())

#distribution of valid and fraud transaction
print(data['Class'].value_counts())      #o/p: here the data is unbalanced as 99% of data is valid which cant be used for accurate prediction

#separating the data for analysis
valid = data[data.Class==0]
fraud = data[data.Class==1]
print(valid.shape)
print(fraud.shape)

#statistical measures of the data
print(valid.Amount.describe())
print(fraud.Amount.describe())

#compare the values for both transaction
print(data.groupby('Class').mean())

'''
perform undersampling
build a sample dataset containing similar distribution of normal transaction and fraudulent transaction
number of fraudulent transaction --> 492
'''

valid_sample = valid.sample(n = 492)

#concatinating two dataframes
new_dataset = pd.concat([valid_sample, fraud], axis = 0)  # axis=o --> dataframes will be added one by one -->axis=0: This parameter specifies the axis along which the concatenation should occur. When axis=0, the DataFrames are stacked vertically, one on top of the other. This means that the rows of valid_sample and fraud will be combined into a single DataFrame.
print(new_dataset.head())
print(new_dataset.tail())

print(new_dataset['Class'].value_counts())

print(new_dataset.groupby('Class').mean()) #here when you compare with previous dataset values, the mean will be almost similar indicating a good sample.

#splitting the data into features and targets
X = new_dataset.drop(columns='Class', axis = 1)
Y = new_dataset['Class']
print(X)
print(Y)

#split the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)  #stratify is used to evenly distribute two classes in x_train and x_test. if u dont mention, the distribution of 0 and 1 can be very different from training and testing data.
print(X.shape, X_train.shape, X_test.shape)

#model traning --> normally we use logistic regression for binary classification problem.
model = LogisticRegression()

#training the logistic regression model with the training data
model.fit(X_train, Y_train)

#model evaluation
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data: ', training_data_accuracy)

#accuracy on test data
X_test_prediction = model.predict(X_test)
X_test_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on testing data: ',X_test_accuracy)

'''
Note: if the accuracy_score of training data is soo diff from accuracy score of testing data, it means that
our model is overfitted or underfitted.
'''

# making a predictive system
input_data = (0,-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction == 0:
    print("Non-Fraudulent Transaction")
else:
    print("Fraudulent Transaction")

