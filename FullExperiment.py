from Loader import LoadTrainingData
import keras.models as models
import keras.layers as layers
from keras import regularizers
from keras.optimizers import rmsprop
import keras
import numpy.random as nr
from Common import SaveModel
from Common import ShowHistoryGraphs
from Prediction import RunPredictions
import keras.utils.np_utils as ku

X_train, X_validate, Y_train, Y_validate = LoadTrainingData(StopAfterN = 0)
inputSize = X_train.shape

nn = models.Sequential()
nn.add(layers.Conv2D(128, kernel_size=(128, 10), activation='relu', input_shape=(inputSize[1],inputSize[2], 1)))
nn.add(layers.MaxPooling2D(pool_size=(1, 2)))
nn.add(layers.Conv2D(64, kernel_size=(1, 10), activation='relu'))
nn.add(layers.MaxPooling2D(pool_size=(1, 2)))
nn.add(layers.Flatten())
nn.add(layers.Dense(32, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)))
nn.add(layers.Dropout(0.5))
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
    epochs = 400, batch_size = 128,
    validation_data = (X_validate, Y_validate), callbacks=callbacks_list)

SaveModel(nn, "model.h5", "model.json")

RunPredictions(nn, resultFilename = 'predictions.csv')

ShowHistoryGraphs(history)
