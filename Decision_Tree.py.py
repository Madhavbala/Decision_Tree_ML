#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# In[3]:


dataset = load_iris()
print("Sepal lenght , Sepal width , Petal length , Petal width \n ")
print(dataset.data)


# In[4]:


print("Output of the dataset.data")
print("Target of dataset     :  ",dataset.target)
print("Shape of the dataset  :  ",dataset.data.shape)


# In[6]:


X_input = pd.DataFrame(dataset.data , columns = dataset.feature_names)
Y_input = dataset.target

print(X_input)
print(Y_input)


# In[7]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(X_input, Y_input, train_size = 0.25,random_state = 0)


# In[8]:


#Finding max_dept value 
accuracy = []
for i in range(1, 10):
    model = DecisionTreeClassifier(max_depth = i, random_state = 0)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    score = accuracy_score(y_test, pred)
    accuracy.append(score)


# In[9]:


plt.figure(figsize=(12, 6))
plt.plot(range(1, 10), accuracy, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Finding best Max_Depth')
plt.xlabel('pred')
plt.ylabel('score')


# In[21]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy',max_depth = 4, random_state = 0)
model.fit(x_train,y_train)


# In[22]:


y_pred = model.predict(x_test)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[23]:


from sklearn.metrics import accuracy_score
print("Accuracy of the Model: {0}%".format(accuracy_score(y_test, y_pred)*100))


# In[ ]:




