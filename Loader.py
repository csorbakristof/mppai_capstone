from matplotlib import image as mp_image
import numpy as np
import os
import csv
import keras.utils.np_utils as ku
from sklearn.model_selection import train_test_split


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

# ---- "public" methods

def LoadTrainingData(StopAfterN = 0):
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

    X = train_images.astype('float32')/255
    Y = ku.to_categorical(np.array(loaded_train_labels)) # Make one-hot encoded numpy array

    X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=0.25)
    X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
    X_validate = X_validate.reshape((X_validate.shape[0],X_validate.shape[1],X_validate.shape[2],1))
    print("X_train: " + str(X_train.shape))
    print("X_validate: " + str(X_validate.shape))
    print("Y_train: " + str(Y_train.shape))
    print("Y_validate: " + str(Y_validate.shape))

    return X_train, X_validate, Y_train, Y_validate



