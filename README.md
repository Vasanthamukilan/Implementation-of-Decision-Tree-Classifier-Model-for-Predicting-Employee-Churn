# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10. Find the accuracy of our model and predict the require values. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:Vasanthamukilan M
RegisterNumber:212222230167
*/
```
```python
import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
data=pd.read_csv("/content/Employee_EX6.csv")
data.head()
data.info()
data.isnull().sum()
data['left'].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```
## Output:
## data.head():
![image](https://github.com/Vasanthamukilan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119559694/8380bb9b-9ed4-4d3c-87ec-76eaffe5bd1e)
## data.info():
![image](https://github.com/Vasanthamukilan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119559694/dc26262f-27c2-463b-8308-f73d74344807)
## data.isnull().sum():
![Screenshot 2024-04-03 134308](https://github.com/Vasanthamukilan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119559694/c4c6029a-dcbd-47cb-8fd5-355833ae3b8b)

## data["left"].value_counts():
![Screenshot 2024-04-03 134316](https://github.com/Vasanthamukilan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119559694/6b342c95-69a5-4bd6-b333-2be422f9858e)

## data.head():
![Screenshot 2024-04-03 134158](https://github.com/Vasanthamukilan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119559694/70fd8df6-a7f3-411e-a0d7-6ca91a8b015d)

## x.head()
![image](https://github.com/Vasanthamukilan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119559694/2d953d7f-1fdb-418a-8f30-cd5e0243614d)

## accuracy:
![Screenshot 2024-04-03 135041](https://github.com/Vasanthamukilan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119559694/0ff4ba1b-cf70-4b18-b6fb-de6db88e3993)

## dt.predict([[0.5,0.8,9,260,6,0,1,2]]):
![Screenshot 2024-04-03 135330](https://github.com/Vasanthamukilan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119559694/ff704fc5-23b4-4045-8295-5b0ae6bed082)
## Decision tree classifier model:
![Screenshot 2024-04-03 135630](https://github.com/Vasanthamukilan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119559694/2dd2f50c-4ba6-427c-a534-62dda91e78ad)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
