# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model
import numpy as np
from Prediction import RunPredictions

nn = load_model('model.h5')
nn.summary()

RunPredictions(nn,'prediction_result.csv')
