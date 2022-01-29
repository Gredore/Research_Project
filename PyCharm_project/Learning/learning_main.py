import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten

from data_prep import *


#NOTE: Configuration MUST be done within data_prep.py before running!
X_train_RDF, y_train_RDF = data_prep()

