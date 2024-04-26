#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[3]:


(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()


# In[4]:


len(X_train)


# In[5]:


len(X_test)


# In[6]:


X_train[0]


# In[7]:


plt.matshow(X_train[0])


# In[8]:


X_train = X_train / 255
X_test = X_test / 255


# In[9]:


X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)


# In[10]:


X_train_flattened[0]


# In[11]:


model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)


# In[12]:


model.evaluate(X_test_flattened, y_test)


# In[13]:


y_predicted = model.predict(X_test_flattened)
y_predicted[0]


# In[14]:


plt.matshow(X_test[0])


# In[15]:


np.argmax(y_predicted[0])


# In[16]:


y_predicted_labels = [np.argmax(i) for i in y_predicted]


# In[17]:


cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
cm


# In[19]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[20]:


model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)


# In[21]:


model.evaluate(X_test_flattened,y_test)


# In[22]:


y_predicted = model.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[26]:


model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='BinaryCrossentropy',
              metrics=['BinaryAccuracy'])

model.fit(X_train_flattened, y_train, epochs=7)


# In[27]:


model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='KLDivergence',
              metrics=['BinaryAccuracy'])

model.fit(X_train_flattened, y_train, epochs=7)


# In[28]:


model.evaluate(X_test_flattened,y_test)



# In[29]:


y_predicted = model.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:




