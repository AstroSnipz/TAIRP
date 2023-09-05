import numpy as np
import pandas as pd   
import matplotlib.pyplot as plt  
import seaborn as sns    
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

titanic_data = pd.read_csv("./titanic_survival_prediction/titanic_data.csv")
print(titanic_data.head())

# number of rows and columns
print(titanic_data.shape)

# getting some more info about the data
titanic_data.info()

# check the number of missing values in each column
print(titanic_data.isnull().sum())

# handling the missing values
# drop the 'Cabin' column from the dataframe
titanic_data = titanic_data.drop(columns='Cabin', axis=1)

# filling the missing values in the 'Age' columns with the mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)  # if u dont mention inplace=True, it will not be saved in original dataframe, it will only be replaced in this particular cell.

# finding the mode value of the 'Embarked' column (i.e, finding the most number of repeated values) -----> here we cant do mean bcoz it is of the form text
print(titanic_data['Embarked'].mode())
print(titanic_data['Embarked'].mode()[0])

# filling the missing values in the embarked column with the mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
print(titanic_data.isnull().sum())

# data analysis
# getting some statistical measures about the data
print(titanic_data.describe())

print(titanic_data['Survived'].value_counts())

# data visualization

    # The sns.set() function is used to set the default aesthetics or styling for Seaborn plots. 

# making a count plot for 'Survived' column
sns.countplot(data=titanic_data, x='Survived')      # x='Survived' specifies that the 'Survived' column should be used as the variable to count and plot on the x-axis.
plt.show()

# making a count plot for 'sex' column
print(titanic_data['Sex'].value_counts())
sns.countplot(data=titanic_data, x='Sex')     
plt.show()
# number of survivors gender wise
sns.countplot(data=titanic_data,x='Sex', hue='Survived')
plt.show()

# making a count plot for 'Pclass' column
print(titanic_data['Pclass'].value_counts())
sns.countplot(data=titanic_data, x='Pclass')     
plt.show()
# number of survivors Pclass wise
sns.countplot(data=titanic_data,x='Pclass', hue='Survived')
plt.show()

# making a count plot for 'SibSp' column
print(titanic_data['SibSp'].value_counts())
sns.countplot(data=titanic_data, x='SibSp')     
plt.show()
# number of survivors SibSp wise   
sns.countplot(data=titanic_data,x='SibSp', hue='Survived')
plt.show()

# encoding the categoricl columns('Sex', 'Embarked')     ----> refer notes
print(titanic_data['Sex'].value_counts())
print(titanic_data['Embarked'].value_counts())
# converting categorical columns
titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
print(titanic_data.head())

# separating features and target
X = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
Y = titanic_data['Survived']
print(X)
print(Y)

# splitting the data into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# model evaluation
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print("accuracy score of training data: ", training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print("accuracy score of test data: ", test_data_accuracy)

# making a predictive system
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")
input_data = (3,0,22.000000,1,0,7.2500,0)

# changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)                   # here we didnt use StandardScalar -->  note video
print(prediction)

if prediction == 0:
    print("The person has not Survived")
else:
    print("The Person has survived")



