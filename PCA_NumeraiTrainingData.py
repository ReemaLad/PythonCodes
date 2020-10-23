#!/usr/bin/env python
# coding: utf-8

# ### Information :
# * Column 1-50 correspond to the features of X or input data, data type: continuous numeric
# * Column 51 correspond to targets fulfilled by Bernie, 1 being Yes (target fulfilled) and 0 being No (Target not fulfilled)
# * data type: discrete continuous

# In[1]:


# importing required libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns


# In[2]:


# importing or loading the dataset 
dataset = pd.read_csv(r"D:\IMS\DataBase\Numerai_Training_Data.csv")


# In[3]:


dataset.head()


# In[4]:


print("---------- Rows & Columns : ---------- \n",dataset.shape)
print("---------- Missing Value Computation ----------")
print(dataset.info())
print("---------- Data Set Statistics ----------")
print(dataset.describe())


# In[5]:


dataset.boxplot()
# Feature 25 & Feature 47???


# # Distributing dataset into two components X and Y 

# In[6]:


X = dataset.drop('target_bernie', axis = 1)
Y = dataset[['target_bernie']]


# In[7]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) 


# In[8]:


train=pd.concat([X_train,y_train],axis=1)


# In[9]:


train.mean(axis=0)


# In[10]:


train.std(axis=0)


# # PCA

# In[11]:


from sklearn.decomposition import PCA
pca = PCA(.95)


# In[12]:


principalComponents = pca.fit_transform(train)


# In[13]:


a=pca.explained_variance_ratio_


# In[14]:


a=pd.DataFrame(data=a,columns = ['Eigen_Values'])
a["PCA"]= range(1,34)


# In[15]:


# Scree Plot
import matplotlib.pyplot as plt
plt.plot( 'PCA', 'Eigen_Values', data=a, linestyle='-', marker='o')
plt.show()


# In[16]:


# Applying PCA function on training and testing set of X component 
from sklearn.decomposition import PCA 
pca = PCA(n_components = 2) # 2 PCA from Scree Plot
  
principalComponents= pca.fit(X_train) # Fit the model with X and apply the dimensionality reduction on X.
transformed_data=principalComponents.fit_transform(X_train)  ### only give data which is transformed ( principal component)


# In[17]:


print("original shape:   ", X_train.shape)
print("transformed shape:", transformed_data.shape)


# In[18]:


print("Explained Variance: ", principalComponents.explained_variance_)
print("Explained Variance Ratio: ", principalComponents.explained_variance_ratio_)


# In[19]:


principalDf = pd.DataFrame(data = transformed_data, columns = ['principal component 1', 'principal component 2'])
principalDf.head()


# In[20]:


train.reset_index(inplace=True)


# In[21]:


finalDf = pd.concat([principalDf, train[['target_bernie']]], axis = 1)


# In[22]:


finalDf=pd.DataFrame(finalDf)
finalDf.head()


# In[23]:


import plotly.express as px
fig = px.scatter(transformed_data, x=0, y=1, color=finalDf['target_bernie'])
fig.show()


# In[24]:


features = ['principal component 1', 'principal component 2']
loadings = principalComponents.components_.T * np.sqrt(principalComponents.explained_variance_)

fig = px.scatter(transformed_data, x=0, y=1, color=finalDf['target_bernie'])

for i, feature in enumerate(features):
    fig.add_shape(
        type='line',
        x0=0, y0=0,
        x1=loadings[i, 0],
        y1=loadings[i, 1]
    )
    fig.add_annotation(
        x=loadings[i, 0],
        y=loadings[i, 1],
        ax=0, ay=0,
        xanchor="center",
        yanchor="bottom",
        text=feature,
    )
fig.show()


# In[25]:


test=pd.concat([X_test,y_test],axis=1)


# In[26]:


test.reset_index(inplace=True)


# In[27]:


X_test_1 = principalComponents.transform(X_test) # Fit the model with X.


# In[28]:


X_test_21 = pd.DataFrame(data = X_test_1, columns = ['principal component 1', 'principal component 2'])
X_test_21.head()


# In[29]:


test_finalDf = pd.concat([X_test_21, test[['target_bernie']]], axis = 1)


# In[30]:


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
ax = sns.lmplot(x="principal component 1", y="principal component 2", data=test_finalDf,
                hue='target_bernie',fit_reg=False,legend=False)
# Move the legend to an empty part of the plot
plt.legend(loc='lower right')


# # FA

# In[31]:


from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer


# In[32]:


# Bartlett Test
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(dataset)
chi_square_value, p_value


# #### Bartlett Test : Non Identify Matrix

# In[33]:


# KMO Test
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(dataset)
kmo_model


# #### The overall KMO for our data is 0.78, i.e. > 50. We can proceed with FA.

# In[34]:


### Choosing Number of Factors
## Determining EigenValues for the same

#Create factor analysis object and perform factor analysis
fa = FactorAnalyzer(rotation='varimax',method="principal")
fa.fit(dataset)
# Check Eigenvalues
eigen_values, vectors = fa.get_eigenvalues()
eigen_values


# In[35]:


dd=pd.DataFrame(eigen_values,columns=["Eigen_Values"])
dd['Factor'] = dd.index
dd.head()


# In[36]:


# Scree Plot
plt.plot('Factor', 'Eigen_Values', data=dd, linestyle='-', marker='o')
plt.show()


# In[37]:


# Create factor analysis object and perform factor analysis
aa = FactorAnalyzer(n_factors=3,rotation='varimax',method="principal")
model = aa.fit(dataset)


# In[39]:


# Correlation Matrix
data_loading = dataset.columns.to_list()
data_loading


# In[ ]:


Correlation = pd.DataFrame(model.corr_,index=data_loading,columns=data_loading)
round(Correlation,3)
Correlation.style.applymap(lambda x: 'background-color : yellow' if x > 0.5 else '')


# In[ ]:


# Factor Loading
factor=["Factor1","Factor2", "Factor3"]
loading=pd.DataFrame(model.loadings_,columns=factor,index=data_loading)
loading=loading.abs()
loading.style.applymap(lambda x: 'background-color : yellow' if x > 0.5 else '')


# In[ ]:


loading.corr()


# In[ ]:


# Communalities
communalities=pd.DataFrame(model.get_communalities(),columns=["communalities"],index=data_loading)
communalities.head()


# In[ ]:


# Uniquess
uniquess=pd.DataFrame(model.get_uniquenesses(),columns=["UnExplained_Variance"],index=data_loading)
uniquess.head()


# In[ ]:


# Eigen Values
header=["Eigen_Value","Perentage of Variance Explained","Cumulative Perentage of Variance Explained"]
variance_Explained=pd.DataFrame(model.get_factor_variance(),columns=factor,index=header)
variance_Explained


# In[ ]:


# Final Transform Data
output = pd.DataFrame(model.fit_transform(dataset))
output.head()


# In[ ]:


features = ["Factor1","Factor2","Factor3"]
loadings = loading.abs() # loadings
fig = px.scatter_3d(output, x=0, y=1, z=2, labels={'0': 'Factor1', '1': 'Factor2', '2': 'Factor3'}, color=dataset['target_bernie'] )
fig.show()


# In[ ]:




