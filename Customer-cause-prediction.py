
# coding: utf-8

# # Phase 1

# In[2]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[3]:


# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# In[4]:


dataset.head()


# In[5]:


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


# In[6]:


X


# In[7]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[8]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[9]:


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


# In[10]:


# Initialising the ANN
classifier = Sequential()


# In[11]:


# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))


# In[12]:


# Adding the hidden layers
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))


# In[13]:


# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


# In[14]:


# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics =['accuracy'])


# In[15]:


# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)


# In[20]:


testdata = pd.DataFrame(X_test)


# In[21]:


#testdata


# In[22]:


traindata = pd.DataFrame(X_train)


# In[23]:


#traindata


# In[24]:


Newdata = testdata.append(traindata, ignore_index= True)


# In[25]:


#Newdata


# In[26]:


new_pred = classifier.predict(Newdata)


# In[27]:


output_df = pd.DataFrame(new_pred, columns = ["11"])


# In[28]:


#output_df


# In[29]:


latest_dataset = pd.concat([Newdata, output_df], axis = 1)


# In[30]:


latest_dataset.head()


# In[31]:


X_new = latest_dataset.iloc[:, 11].values
y_new = latest_dataset.iloc[:, 0:11].values


# In[32]:


X_new


# In[33]:


X_new1 = X_new.reshape(10000,1)


# In[34]:


X_new1


# In[35]:


X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_new1, y_new, test_size = 0.5, random_state= 1)


# In[36]:


X_new_train


# In[37]:


y_new_train


# # Phase 2

# # Using Multiple Regression

# In[38]:


from sklearn.linear_model import LinearRegression


# In[39]:


reg = LinearRegression()


# In[40]:


reg.fit(X_new_train, y_new_train)


# In[41]:


reg.predict(X_new_test[:1,:])


# In[42]:


reg.score(X_new_test, y_new_test)


# In[43]:


X_new_train.shape


# In[44]:


y_new_train.shape


# # Using ANN

# In[53]:


model = Sequential()
# the shape of one training example is
input_shape = X_new_train[0].shape
model.add(Dense(units=11, activation='relu', input_shape=input_shape))

# Hidden Layers
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=11, activation='relu'))

# Output Layers
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))
model.add(Dense(units=11, activation='softmax'))

# Compile the ann
model.compile(optimizer='rmsprop', loss='binary_crossentropy', 
metrics=['accuracy'])


# In[54]:


model.fit(X_new_train, y_new_train, batch_size = 10, epochs=50)


# In[47]:


model.predict(X_new_test[0, :])

