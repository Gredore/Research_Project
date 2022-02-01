import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from keras.callbacks import EarlyStopping
import sklearn
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt

from data_prep import *

num_folds = 10
epochs = 200
early_stopping_patience = 5000
NCOMPONENTS = 40 #For Principal Component Analysis

#NOTE: Configuration MUST be done within data_prep.py before running!
X_train_RDF, y_train_RDF, class_weights_dict, num_stable_RDFs, num_unstable_RDFs = data_prep()

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

kfold = KFold(n_splits=num_folds, shuffle=True)

#val_accuracy_summed = np.zeros([1, epochs])

fold_no = 1
for train, test in kfold.split(X_train_RDF, y_train_RDF):

    pca = PCA(n_components=NCOMPONENTS)
    X_pca_train = pca.fit_transform(X_train_RDF[train][:,:,0])
    X_pca_test = pca.transform(X_train_RDF[test][:,:,0])

    model = Sequential()

    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=([X_pca_train.shape[1], 1])))
    #model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(Flatten())
    #model.add(Dense(30, activation='softmax'))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(f'Training for fold {fold_no} ...')

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stopping_patience)

    #history = model.fit(X_train_RDF[train], y_train_RDF[train],validation_data=(X_train_RDF[test], y_train_RDF[test]),class_weight=class_weights_dict, epochs=epochs, verbose=0, callbacks=[es])
    history = model.fit(X_pca_train, y_train_RDF[train], validation_data=(X_pca_test, y_train_RDF[test]),
                        class_weight=class_weights_dict, epochs=epochs, verbose=0, callbacks=[es])

    #val_accuracy_summed = val_accuracy_summed + np.array(history.history['val_accuracy'])

    # Generate generalization metrics
    #scores = model.evaluate(X_train_RDF[test], y_train_RDF[test], verbose=0)
    scores = model.evaluate(X_pca_test, y_train_RDF[test], verbose=0)

    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    fold_no = fold_no + 1

print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)/num_folds})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')
print("Weighted random guessing would give accuracy: ", num_stable_RDFs/(num_stable_RDFs+num_unstable_RDFs))
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.plot(val_accuracy_summed)
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')

#plt.plot(val_accuracy_summed[0,:])
#plt.show()