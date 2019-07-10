#!/usr/bin/env python
# coding: utf-8

# # My capstone project for MPP AI

# In[1]:


import keras.utils.np_utils as ku
import keras.models as models
import keras.layers as layers
from keras import regularizers
from keras.optimizers import rmsprop
import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')


# ## Initializing and loading

# In[ ]:


import os
import csv
from matplotlib import image as mp_image
#get_ipython().magic(u'matplotlib inline')

def loadImages(folder, stopAfter10 = False):
    images = []
    n = 0
    for f in os.listdir(folder):
        n=n+1
        if (n>10 and stopAfter10):
            break
        fileWithPath = os.path.join(folder, f)
#        print("Loading image: " + fileWithPath)
        images.append( mp_image.imread(fileWithPath) )
    print("images: len=" + str(len(images))+ ", image shape: " + str(images[0].shape))    
    return images
    
def loadLabelsFromCsvWithHeader(filename):
    reader = csv.DictReader(open(filename),fieldnames=["file_id", "accent"])
    labels = []
    isFirst = True
    for row in reader:
        if (isFirst):
            isFirst = False
        else:
            labels.append(row["accent"])
    return labels

StopAfter10 = False

loaded_train_images = np.stack( loadImages("data/train", stopAfter10=StopAfter10), axis=0)
loaded_train_labels = loadLabelsFromCsvWithHeader("data/train_labels.csv")
if (StopAfter10):
    loaded_train_labels = loaded_train_labels[0:10]
    print("WARNING! ONLY 10 IMAGES WERE LOADED!")

train_images = loaded_train_images[:, :, :, 0]

plt.imshow(loaded_train_images[0,:,:,:])
plt.show()


# In[123]:


from sklearn.model_selection import train_test_split

X = train_images.astype('float32')/255
Y = ku.to_categorical(np.array(loaded_train_labels)) # Make one-hot encoded numpy array
print("X shape: " + str(X.shape))

X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=0.25)
print("X_train shape: " + str(X_train.shape))
print("X_train shape: " + str(X_train.shape))
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
X_validate = X_validate.reshape((X_validate.shape[0],X_validate.shape[1],X_validate.shape[2],1))
print("X_train shape: " + str(X_train.shape))

print("X_train: " + str(X_train.shape))
print("X_validate: " + str(X_validate.shape))
print("Y_train: " + str(Y_train.shape))
print("Y_validate: " + str(Y_validate.shape))

print(Y_train)


# ## Data preprocessing

# In[124]:

X_train.shape

# ## Creating the DNN

# In[125]:


inputSize = X_train.shape
print(inputSize)

nn = models.Sequential()
#nn.add(layers.Dense(64, activation = 'relu', input_shape = (inputSize, )))
#nn.add(layers.Dense(3, activation = 'softmax'))

nn.add(layers.Conv2D(64, kernel_size=(128, 10), activation='relu', input_shape=(inputSize[1],inputSize[2], 1)))
nn.add(layers.MaxPooling2D(pool_size=(1, 10)))
nn.add(layers.Conv2D(32, kernel_size=(1, 10), activation='relu'))
nn.add(layers.MaxPooling2D(pool_size=(1, 3)))
#nn.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
nn.add(layers.Flatten())
nn.add(layers.Dense(64, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)))
nn.add(layers.Dense(32, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)))
#nn.add(layers.Dropout(0.5))
nn.add(layers.Dense(3, activation = 'softmax'))

nn.summary()
nn.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# ## Training the network, saving history

# In[126]:


nr.seed(1025)

# batch size was 128
history = nn.fit(X_train, Y_train, 
    epochs = 500, batch_size = 128,
    validation_data = (X_validate, Y_validate))


# ## Evaluation
# Visualization of the history, checking for overfitting...
# Estimating accuracy
# Checking confusion matrix

# In[127]:


# Using the validation set, we do not need "nn.evaluate(test_images, test_labels)"...
fig = plt.figure(figsize=(12, 6))
ax1=plt.subplot(1, 2, 1)

def plot_loss(history):
    train_loss = history.history['loss']
    test_loss = history.history['val_loss']
    x = list(range(1, len(test_loss) + 1))
    plt.plot(x, test_loss, color = 'red', label = 'test loss')
    plt.plot(x, train_loss, label = 'traning loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    
plot_loss(history) 
plt.show()

ax1=plt.subplot(1, 2, 2)

def plot_accuracy(history):
    train_acc = history.history['acc']
    test_acc = history.history['val_acc']
    x = list(range(1, len(test_acc) + 1))
    plt.plot(x, test_acc, color = 'red', label = 'test accuracy')
    plt.plot(x, train_acc, label = 'training accuracy')  
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')  
    plt.legend(loc='lower right')
    
plot_accuracy(history)
plt.show()


# In[84]:


#from keras import metrics
#predictions = nn.predict(X_validate)
#matrix = metrics.confusion_matrix(Y_validate.argmax(axis=1), Y_pred.argmax(axis=1))


# In[ ]:


## Evaluating new data


# In[ ]:


#loaded_test_images = np.stack( loadImages("data/test"), axis=0)
#test_images = loaded_test_images[:, :, :, 0]
#X_test = imageSetToArray(test_images)
#Y_test = ku.to_categorical(np.array(loaded_train_labels))

