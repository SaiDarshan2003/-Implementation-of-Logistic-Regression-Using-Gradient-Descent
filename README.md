# Implementation of Logistic Regression Using Gradient Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: 
RegisterNumber:  
import numpy as np
import matplotlib.pyplot as plt
import  pandas as pd
datasets=pd.read_csv('Social_Network_Ads (1).csv')
X=datasets.iloc[:,[2,3]].values
Y=datasets.iloc[:,4].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_x
StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)
from sklearn.linear_model import LogisticRegression
Regressor=LogisticRegression(random_state=0)
Regressor.fit(X_train,Y_train)
y_pred=Regressor.predict(X_test)
y_pred
from sklearn_metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)
cm
from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,y_pred)
accuracy
recall_sensitivity=metrics.recall_score(Y_test,y_pred,pos_label=1)
recall_specificity=metrics.recall_score(Y_test,y_pred,pos_label=0)
recall_sensitivity,recall_specificity
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
X_set,Y_Set=X_train,Y_train
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,Regressor.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(('red','blue')))
plt.xlim(X1.min(),X2.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_set[Y_Set==j,0],X_set[Y_Set==j,1],c=ListedColormap(('black','purple'))(i),label=j)
plt.title('Logistic Regression(Training Set)',color='yellow')
plt.xlabel('Age',color='yellow')
plt.ylabel('Estimated Salary',color='yellow')
plt.legend()
plt.show()
*/
```

## Output:
![logistic regression using gradient descent](sam.png)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

