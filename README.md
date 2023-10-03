![l4 5](https://github.com/Kowsalyasathya/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118671457/71c8b37a-872b-4a24-84e0-392bb5a6eec5)![l4 4](https://github.com/Kowsalyasathya/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118671457/00091985-bc46-4056-99a1-c41a81c8d263)# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: KOWSALYA M
RegisterNumber:  212222230069
```
```
import pandas as pd
df=pd.read_csv('/content/Placement_Data(1).csv')
df.head()

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)#removes the specified row or column
df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
df1

x=df1.iloc[:,:-1]
x

y=df1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion =confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics  import classification_report
classification_report = classification_report(y_test,y_pred)
print(classification_report)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
### Placements data Salary data :
![l4 1](https://github.com/Kowsalyasathya/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118671457/b54fdcb7-a837-45b0-836d-66ab4caff108)
### Checking null values:
![l2](https://github.com/Kowsalyasathya/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118671457/48019c7d-8046-4593-a45b-c30f56cd672a)
### Print data:
![l4 3](https://github.com/Kowsalyasathya/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118671457/5d88ee70-7b74-488c-8767-2a1eaf16717e)
### Data-Status:
![l4 4](https://github.com/Kowsalyasathya/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118671457/cfbb91b5-ad62-453d-ad84-a75a28965ef3)

![l4 5](https://github.com/Kowsalyasathya/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118671457/8fc107c6-5aaa-417f-9d27-207b532a6d40)

![l4 6](https://github.com/Kowsalyasathya/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118671457/a6cc37cd-63ee-47de-9bf3-c306c7796444)
### Y_prediction array:
![l4 7](https://github.com/Kowsalyasathya/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118671457/d8fc4c0f-6ed2-4264-b396-d34b727d9f25)
### Accuracy value:
![l4 8](https://github.com/Kowsalyasathya/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118671457/6511b502-e164-421e-872e-a9ab2923f7f4)

### Confusion array:
![l4 9](https://github.com/Kowsalyasathya/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118671457/1c66e62e-76fb-4e98-bfe9-27022e45736b)
### Prediction of LR:
![l4](https://github.com/Kowsalyasathya/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118671457/75851c26-bfad-463a-92de-b440509805e8)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
