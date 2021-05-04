#!/usr/bin/env python
# coding: utf-8

# # Name :- Prathamesh S. Kumbhar
# # ML MAJOR PROJECT

# # TASK:
# 
# # PREDICT THE TOTAL SCORE OF AN IPL MATCH
# #DATASET:
# 
# https://drive.google.com/file/d/1ldUmBu7_IF-1B_m5YYM3fejzs1C11OYS/view?usp=sharing
# 

# 1.Handle missing values.<br>
# 2.Drop the unnecessary columns.<br>
# 3.Convert the categorical string columns to numerical columns, by using one-hot encoding.<br>
# 4.Perform feature scaling (if necessary).<br>
# 5.Build a model on the “total” column, using a RandomForestRegressor.<br>
# 6.Calculate the score.<br>
# 7.Predict on a new set of features (you can create a new dataset, having just 1 row, with your preferred feature values).<br>

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv("C:/Users/LIZA MARY MATHEWS/Downloads/ipl2017.csv")
df.head()


# # Dropping unneccesary columns

# In[ ]:


df = df.drop(['mid','date'],axis=1)
df.head()


# In[ ]:


df.describe()


# In[ ]:





# # Checking unique values

# In[ ]:


a=df['venue'].unique()
print(len(a))
a.sort()
a


# In[ ]:


b=df['bat_team'].unique()
print(b)
print(len(b))


# In[ ]:


c=df['bowl_team'].unique()
print(c)
print(len(c))


# In[ ]:


df = df.replace("Rising Pune Supergiants","Rising Pune Supergiant")


# In[ ]:


b=df['bat_team'].unique()
print(len(b))
b.sort()
print(b)
c=df['bowl_team'].unique()
print(len(c))
c.sort()
print(c)


# In[ ]:


d=df['batsman'].unique()
print(len(d))
d.sort()
d


# In[ ]:


e=df['bowler'].unique()
print(len(e))
e.sort()
e


# In[ ]:





# # Handling missing values (if any)

# In[ ]:


X = df.drop('total',axis=1)
print('Shape of X : ',X.shape)
print('Type of X : ',type(X))


# In[ ]:


y = df['total']
print('Shape of y : ',y.shape)
print('Type of y : ',type(y))


# In[ ]:


df.info()


# In[ ]:


# No missing values found


# # One hot encoding

# In[ ]:


X = pd.get_dummies(X,columns=['venue','bat_team','bowl_team','batsman','bowler'])
X.head()


# In[ ]:


X.columns


# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt
plt.hist(df['venue'])
plt.show()
plt.hist(df['bat_team'])
plt.show()
plt.hist(df['bowl_team'])
plt.show()
plt.hist(df['batsman'])
plt.show()
plt.hist(df['bowler'])
plt.show()
plt.hist(df['runs'])
plt.show()
plt.hist(df['wickets'])
plt.show()
plt.hist(df['overs'])
plt.show()


# In[ ]:





# # Splitting into training and testing sets

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2,random_state=42)


# In[ ]:





# # Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler=StandardScaler()


# In[ ]:


X_train=scaler.fit_transform(X_train)


# In[ ]:


X_test=scaler.transform(X_test)


# In[ ]:





# # Random Forest Regressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


model=RandomForestRegressor()


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


model.score(X_test,y_test)


# In[ ]:


model.score(X_train,y_train)


# In[ ]:





# # Predicting on new dataset

# In[ ]:


df_trial= pd.read_csv('C:/Users/LIZA MARY MATHEWS/Desktop/dataset/ipl_small.csv')


# In[ ]:


df_trial.head()


# In[ ]:


model.predict(df_trial)


# In[ ]:




