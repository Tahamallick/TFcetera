#!/usr/bin/env python
# coding: utf-8

# # Small Image Classification Using Simple Aritifical Neural Network: GPU Benchmarking
# 

# In[1]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


tf.config.experimental.list_physical_devices()


# In[3]:


tf.__version__


# In[4]:


tf.test.is_built_with_cuda()


# #Load the dataset
# 

# In[5]:


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[6]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[7]:


train_images.shape


# In[8]:


plt.imshow(train_images[0])


# In[9]:


train_labels[0]


# In[10]:


class_names[train_labels[0]]


# In[11]:


plt.figure(figsize=(3,3))
for i in range(5):
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i]])
    plt.show()


# In[12]:


train_images_scaled = train_images / 255.0
test_images_scaled = test_images / 255.0


# In[13]:


def get_model(hidden_layers=1):
  layers = []

  layers.append(keras.layers.Flatten(input_shape=(28, 28))) 

  for _ in range(hidden_layers):
    layers.append(keras.layers.Dense(units=128, activation='relu'))  

  layers.append(keras.layers.Dense(units=10, activation='softmax'))

  model = keras.Sequential(layers)

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model


# In[14]:


model = get_model(1)
model.fit(train_images_scaled, train_labels, epochs=5)


# In[15]:


model.predict(test_images_scaled)[2]


# In[16]:


test_labels


# In[17]:


tf.config.experimental.list_physical_devices() 


# # 5 Epochs performance comparison for 1 hidden layer
# 

# In[19]:


get_ipython().run_cell_magic('timeit', '-n1 -r1', "with tf.device('/CPU:0'):\n    model = get_model(1)\n    model.fit(train_images_scaled, train_labels, epochs=5)\n")


# In[20]:


get_ipython().run_cell_magic('timeit', '-n1 -r1', "with tf.device('/GPU:0'):\n    model = get_model(1)\n    model.fit(train_images_scaled, train_labels, epochs=5)\n")


# # 5 Epocs performance comparison with 5 hidden layers
# 

# In[21]:


get_ipython().run_cell_magic('timeit', '-n1 -r1', "with tf.device('/CPU:0'):\n    model = get_model(5)\n    model.fit(train_images_scaled, train_labels, epochs=5)\n")


# In[22]:


get_ipython().run_cell_magic('timeit', '-n1 -r1', "with tf.device('/GPU:0'):\n    model = get_model(5)\n    model.fit(train_images_scaled, train_labels, epochs=5)\n")


# In[ ]:




