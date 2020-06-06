#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# The HR Team has collected extensive data on their employees and approached you to develop a model that could predict
# which employees are more likely to quit.


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


employee_df = pd.read_csv('Human_Resources.csv')


# In[6]:


employee_df


# In[7]:


employee_df.head()


# In[8]:


employee_df.info()


# In[9]:


employee_df.describe()


# In[10]:


# Data Visualization


# In[11]:


employee_df['Attrition'] = employee_df['Attrition'].apply(lambda x:1 if x=='Yes' else 0)


# In[12]:


employee_df.head()


# In[13]:


employee_df['OverTime'] = employee_df['OverTime'].apply(lambda x:1 if x=='Yes' else 0)
employee_df['Over18'] = employee_df['Over18'].apply(lambda x:1 if x=='Y' else 0)


# In[14]:


employee_df.head()


# In[15]:


#finding missing value


# In[16]:


sns.heatmap(employee_df.isnull(), yticklabels = False, cbar = False,cmap = 'Blues')


# In[17]:


#ploting histogram of dataframe
employee_df.hist(bins = 30 , figsize = (20,20), color = 'g')


# In[18]:


# here we are dropping 'EmployeeCount','Standardhours','over18',since they do not change from one empolyee to other
#drop 'EmployeeNumber' as well
employee_df.drop(['EmployeeCount','StandardHours','Over18','EmployeeNumber'],axis=1,inplace = True)


# In[19]:


employee_df.head()


# In[20]:


# how many employee who stayed in company
left_df = employee_df[employee_df['Attrition'] == 1]
stayed_df = employee_df[employee_df['Attrition'] == 0]


# In[21]:


#counting number who left and stayed
print('Total = ',len(employee_df))
print('Number of employee who left = ', len(left_df))
print('Number of employee who stayed = ',len(stayed_df))


# In[22]:


left_df.describe()


# In[23]:


stayed_df.describe()


# In[24]:


correlations = employee_df.corr()
f,ax = plt.subplots(figsize = (20,20))
sns.heatmap(correlations , annot = True)


# In[25]:


plt.figure(figsize = [25,12])
sns.countplot(x = 'Age',hue = 'Attrition',data = employee_df)


# In[26]:


plt.figure(figsize = [20,30])

plt.subplot(411)
sns.countplot(x = 'JobRole',hue = 'Attrition',data = employee_df)

plt.subplot(411)
sns.countplot(x = 'MaritalStatus',hue = 'Attrition',data = employee_df)

plt.subplot(411)
sns.countplot(x = 'JobInvolvement',hue = 'Attrition',data = employee_df)

plt.subplot(411)
sns.countplot(x = 'JobLevel',hue = 'Attrition',data = employee_df)


# In[27]:


#KDE is used for visualization the probability density of continous variables
#kde describe  the probability density at different values in a continous variable.


# 

# In[28]:


#gnder vs Monthly Income
sns.boxplot(x = 'MonthlyIncome',y = 'Gender',data = employee_df)


# In[29]:


#job role vs Monthly income
plt.figure(figsize = (15,10))
sns.boxplot(x = 'MonthlyIncome',y = 'JobRole',data = employee_df)


# In[30]:


#create testing and training dataset and perform data cleaning.


# In[31]:


employee_df.head(3)


# In[32]:


# cat stand for categorical data
X_cat = employee_df[['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus']]
X_cat


# In[33]:


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()


# In[34]:


X_cat.shape


# In[35]:


X_cat = pd.DataFrame(X_cat)


# In[36]:


X_cat


# In[37]:


## note that we dropped the target 'Atrittion'
X_numerical = employee_df[['Age', 'DailyRate', 'DistanceFromHome',	'Education', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',	'JobLevel',	'JobSatisfaction',	'MonthlyIncome',	'MonthlyRate',	'NumCompaniesWorked',	'OverTime',	'PercentSalaryHike', 'PerformanceRating',	'RelationshipSatisfaction',	'StockOptionLevel',	'TotalWorkingYears'	,'TrainingTimesLastYear'	, 'WorkLifeBalance',	'YearsAtCompany'	,'YearsInCurrentRole', 'YearsSinceLastPromotion',	'YearsWithCurrManager']]
X_numerical


# In[38]:


X_all = pd.concat([X_cat,X_numerical],axis = 1)


# In[39]:


X_all


# In[40]:


from sklearn.preprocessing import MinMaxScaler


# In[41]:


scaler = MinMaxScaler()
X = scaler.fit_transform(X_all)


# In[42]:


X


# In[43]:


y = employee_df['Attrition']
y


# In[44]:


#using logestic regression


# In[47]:


from sklearn.model_selection import train_test_split


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# In[49]:


X_train.shape


# In[50]:


X_test.shape


# In[51]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[53]:


model = LogisticRegression()


# In[54]:


model.fit(X_train,y_train)


# In[55]:


y_pred = model.predict(X_test)


# In[56]:


y_pred


# In[57]:


from sklearn.metrics import confusion_matrix,classification_report


# In[58]:


print('Accuracy {} %'.format(100 * accuracy_score(y_pred,y_test)))


# In[59]:


cm = confusion_matrix(y_pred,y_test)
sns.heatmap(cm,annot = True)


# In[60]:


print(classification_report(y_test,y_pred))


# In[61]:


# now training with Random Forest Classifier


# In[62]:


from sklearn.ensemble import RandomForestClassifier


# In[63]:


model = RandomForestClassifier()


# In[64]:


model.fit(X_train,y_train)


# In[65]:


y_pred = model.predict(X_test)


# In[66]:


cm = confusion_matrix(y_pred,y_test)
sns.heatmap(cm,annot = True)


# In[67]:


print(classification_report(y_test,y_pred))

