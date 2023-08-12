"""**IMPORTING LIBRARIES**"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split

"""LOAD THE DATASET """

data=pd.read_csv("./diabetes.csv")
print("******************************** Data ************************************ \n")
print(data)

"""**SUMMARIZE THE DATASET**"""
print("\n\n********************************** Top 5 rows in Datset ********************************** \n")
print(data.head())             #it displays top 5 rows in the dataset
print("\n\n********************************** Description About dataset  ********************************** \n")
print(data.describe())     #it gives description about dataset
print("\n\n********************************** Bottom 5 rows in Datset ********************************** \n")
print(data.tail()  )       #it displays last 5 rows in the dataset
print("\n\n********************************** shape of Datset(no of rows,no of columns) ********************************** \n")
print(data.shape)          #it shows(no of rows,no of columns)

"""**SEGREGEATE DATA INTO INPUT AND OUPUT**"""

x=data.drop(columns="diabetes")         #input data
print("\n\n********************************** Input label data== ********************************** \n")
print(x)
print("\n\n********************************** Output label data== ********************************** \n")
y=data.diabetes                       #output label
print(y)

"""PREPROCESS THE INPUT(NORMALIZATION)"""

scale=StandardScaler()
x_scale=scale.fit_transform(x)

"""**TRAIN THE MODEL USING SVM MODEL**"""

x_train,x_test,y_train,y_test=train_test_split(x_scale,y,test_size=0.2)
print("\n********************************** Shape of Traiing data== **********************************")
print(x_train.shape)
print("\n********************************** Shape of Testing data== **********************************")
print(x_test.shape)
print("\n")

svmclassifier=svm.SVC(kernel="linear")
svmclassifier.fit(x_train, y_train)

"""**FINDING ACCURACY OF TRAINING DATA,TEST DATA**"""

# accuracy score on the training data
x_train_prediction = svmclassifier.predict(x_train)
training_accuracy = accuracy_score(x_train_prediction, y_train)
accuracy=100*training_accuracy
print("accuracy of trianing data={0}".format(accuracy))

# accuracy score on the training data
x_test_prediction = svmclassifier.predict(x_test)
testing_accuracy = accuracy_score(x_test_prediction, y_test)
accuracy=100*testing_accuracy
print("accuracy of testing  data={0}".format(accuracy))
print("\n")

"""**PREDICTING /TESTING THE DATA**"""

#input= (5,166,72,19,175,25.8,0.587,51,1.09)
input=(6,148,72,35,0,33.6,0.627,50,1.3790)
input_array = np.asarray(input)
input_reshape = input_array.reshape(1,-1)
std_data = scale.transform(input_reshape)
prediction = svmclassifier.predict(std_data)

"""**CHECK WHETHER PERSON HAVE DIABETICS OR NOT**"""

if (prediction[0] == 0):
  print('THE PERSON IS NOT DIABETIC')
else:
  print('THE PERSON IS DIABETIC')