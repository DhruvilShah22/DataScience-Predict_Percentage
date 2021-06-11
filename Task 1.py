#!/usr/bin/env python
# coding: utf-8

# # TASK 1
# To Predict the percentage of marks of the students based on the number of hours they studied

# # Project Done By -- Dhruvil Shah 
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# In[2]:


data = pd.read_csv ('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
data.head(25)


# In[3]:


data.isnull == True


# In[4]:


sns.set_style('darkgrid')
sns.scatterplot(y= data['Scores'], x= data['Hours'])
plt.title('Marks Vs Study Hours',size=25)
plt.ylabel('Marks Percentage', size=20)
plt.xlabel('Hours Studied', size=20)
plt.show()


# In[5]:


sns.regplot(x= data['Hours'], y= data['Scores'])
plt.title('Regression Plot',size=25)
plt.ylabel('Marks Percentage', size=20)
plt.xlabel('Hours Studied', size=20)
plt.show()
print(data.corr())


# In[6]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# In[7]:


regression = LinearRegression()
regression.fit(train_X, train_y)
print("Model Has Been Trained ")


# In[8]:


pred_y = regression.predict(val_X)
prediction = pd.DataFrame({'Hours': [i[0] for i in val_X], 'Predicted Marks': [k for k in pred_y]})
prediction

# Comparing the Predicted Marks with the Actual Marks

compare_scores = pd.DataFrame({'Actual Marks': val_y, 'Predicted Marks': pred_y})
compare_scores
# In[9]:


plt.scatter(x=val_X, y=val_y, color='blue')
plt.plot(val_X, pred_y, color='Black')
plt.title('Actual vs Predicted', size=25)
plt.ylabel('Marks Percentage', size=20)
plt.xlabel('Hours Studied', size=20)
plt.show()


# In[11]:


print('Mean absolute error is :',mean_absolute_error(val_y,pred_y))


# In[12]:


hours = [9.25]
answer = regression.predict([hours])
print("According to the regression model if a student studies for 9.25 hours a day he/she is likely to score = {} marks".format(round(answer[0],3)))

