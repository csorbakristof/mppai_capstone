#!/usr/bin/env python
# coding: utf-8

# # My capstone project for MPP AI
import keras.utils.np_utils as ku
import keras.models as models
import keras.layers as layers
from keras import regularizers
from keras.optimizers import rmsprop
from matplotlib import image as mp_image
import numpy as np
import os
import csv
import numpy.random as nr
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
#get_ipython().magic(u'matplotlib inline')

def loadImages(folder, stopAfterN = 0):
    images = []
    file_ids = []
    n = 0
    for f in os.listdir(folder):
        n=n+1
        if (n>stopAfterN and stopAfterN>0):
            break
        fileWithPath = os.path.join(folder, f)
        images.append( mp_image.imread(fileWithPath) )
        file_ids.append(f.split(".")[0])
    print("images: len=" + str(len(images))+ ", image shape: " + str(images[0].shape))    
    return images, file_ids
    
def loadLabelsFromCsvWithHeader(filename):
    reader = csv.DictReader(open(filename),fieldnames=["file_id", "accent"])
    labels = []
    file_ids = []
    isFirst = True
    for row in reader:
        if (isFirst):
            isFirst = False
        else:
            labels.append(row["accent"])
            file_ids.append(row["file_id"])
    return labels, file_ids

StopAfterN = 0

images, images_file_ids = loadImages("data/train", stopAfterN=StopAfterN)
loaded_train_images = np.stack( images, axis=0)
loaded_train_labels, label_file_ids = loadLabelsFromCsvWithHeader("data/train_labels.csv")
if (StopAfterN>0):
    loaded_train_labels = loaded_train_labels[0:StopAfterN]
    label_file_ids = label_file_ids[0:StopAfterN]
    print("WARNING! ONLY "+str(StopAfterN)+" IMAGES WERE LOADED!")

if (len(images_file_ids) != len(label_file_ids)):
    print("ERROR: IMAGE AND LABEL LIST LENGTH DOES NOT MATCH!")

for i in range(len(images_file_ids)):
    if (images_file_ids[i] != label_file_ids[i]):
        print("ERROR: FILE_ID MISMATCH!!!")

train_images = loaded_train_images[:, :, :, 0]

#plt.imshow(loaded_train_images[0,:,:,:])
#plt.show()


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




inputSize = X_train.shape
print(inputSize)

nn = models.Sequential()

nn.add(layers.Conv2D(128, kernel_size=(128, 10), activation='relu', input_shape=(inputSize[1],inputSize[2], 1)))
nn.add(layers.MaxPooling2D(pool_size=(1, 2)))
nn.add(layers.Conv2D(64, kernel_size=(1, 10), activation='relu'))
nn.add(layers.MaxPooling2D(pool_size=(1, 2)))
#nn.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
nn.add(layers.Flatten())
nn.add(layers.Dense(32, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)))
#nn.add(layers.Dropout(0.5))
nn.add(layers.Dense(3, activation = 'softmax'))

nn.summary()
nn.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# ## Training the network, saving history

checkpointfilename = "earlystopcheckpoint.hdf5"
callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor = 'val_loss', # Use accuracy to monitor the model
        patience = 100 # Stop after N steps with lower accuracy
    ),
    keras.callbacks.ModelCheckpoint(
        filepath = checkpointfilename, # file where the checkpoint is saved
        monitor = 'val_loss', # Don't overwrite the saved model unless val_loss is worse
        save_best_only = True # Only save model if it is the best
    )
]

nr.seed(1025)

# batch size was 128
history = nn.fit(X_train, Y_train, 
    epochs = 500, batch_size = 128,
    validation_data = (X_validate, Y_validate), callbacks=callbacks_list)

# --- save model and weights
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# serialize model to JSON
nn_json = nn.to_json()
with open("model.json", "w") as json_file:
    json_file.write(nn_json)
# serialize weights to HDF5
nn.save_weights("model.h5")
print("Saved model to disk")

# ----------------- run predictions

images, file_ids = loadImages("data/test", stopAfterN=StopAfterN)
loaded_test_images = np.stack( images, axis=0)
test_images = loaded_test_images[:, :, :, 0]
X_test = test_images.astype('float32')/255

X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))

print("Now creating predictions to test data...")
Y_pred = nn.predict_classes(X_test)

with open('predictions.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    filewriter.writerow(["file_id", "accent"])
    for i in range(len(file_ids)):
        filewriter.writerow([file_ids[i].split(".")[0], Y_pred[i]])

print("predictions.csv ready")


# -------------  Evaluation
# Visualization of the history, checking for overfitting...
# Estimating accuracy
# Checking confusion matrix

if (StopAfterN>0):
    print("WARNING! ONLY "+str(StopAfterN)+" IMAGES WERE LOADED (for train and for predictionas well)!")


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


