import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPool1D
from keras.callbacks import EarlyStopping
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt

from data_prep import *

num_folds = 10
epochs = 200
early_stopping_patience = 5000

#NOTE: Configuration MUST be done within data_prep.py before running!
X_train_RDF_unit, y_train_RDF_not_cat, class_weights_dict, num_stable_RDFs, num_unstable_RDFs = data_prep(use_catagorical_y=False, atomic_weighting="unit")
X_train_RDF_electroneg, _, _, _, _ = data_prep(use_catagorical_y=False, atomic_weighting="electroneg")
X_train_RDF_vdW, _, _, _, _ = data_prep(use_catagorical_y=False, atomic_weighting="vdW")

X_train_RDF_unit_minus_electroneg = X_train_RDF_unit - X_train_RDF_electroneg
X_train_RDF_unit_minus_vdW = X_train_RDF_unit - X_train_RDF_vdW

y_train_RDF = to_categorical(y_train_RDF_not_cat)  # Required after kfold as statifiedkfold requires non-categorical unlike normal kfold.

X_train_RDF = np.concatenate((X_train_RDF_unit, X_train_RDF_unit_minus_electroneg, X_train_RDF_unit_minus_vdW), axis=2)
#X_train_RDF = X_train_RDF_vdW

Repeats_of_shuffled_splits = 1

cf_matrix_repeats = np.zeros([2, 2, Repeats_of_shuffled_splits*num_folds])
MCCs = np.zeros([Repeats_of_shuffled_splits*num_folds])
val_accuracy_summed = np.zeros(epochs)

fold_counter_including_repeats = 0

for i in range (0, Repeats_of_shuffled_splits):
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)


    fold_no = 1

    for train, test in kfold.split(X_train_RDF, y_train_RDF_not_cat):

        model = Sequential()

        model.add(Conv1D(16, kernel_size=3, activation='relu', input_shape=(X_train_RDF[train].shape[1:])))
        model.add(MaxPool1D(pool_size=3))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print(f'Training for fold {fold_no} ...')

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stopping_patience)

        #history = model.fit(X_train_RDF[train], y_train_RDF[train],validation_data=(X_train_RDF[test], y_train_RDF[test]),class_weight=class_weights_dict, epochs=epochs, verbose=0, callbacks=[es])
        history = model.fit(X_train_RDF[train], y_train_RDF[train], validation_data=(X_train_RDF[test], y_train_RDF[test]),
                            class_weight=class_weights_dict, epochs=epochs, verbose=0, callbacks=[es])

        val_accuracy_summed = val_accuracy_summed + np.array(history.history['val_accuracy'])

        # Generate generalization metrics
        #scores = model.evaluate(X_train_RDF[test], y_train_RDF[test], verbose=0)
        scores = model.evaluate(X_train_RDF[test], y_train_RDF[test], verbose=0)


        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')

        y_test = np.argmax(y_train_RDF[test], axis=1)
        y_pred_strengths = model.predict(X_train_RDF[test])
        y_pred = np.argmax(y_pred_strengths, axis=1)
        #print(y_test)
        #print(y_pred)

        current_cf_matrix = confusion_matrix(y_test, y_pred)
        current_cf_matrix = current_cf_matrix / (current_cf_matrix[1, 1])
        cf_matrix_repeats[:, :, fold_counter_including_repeats] = current_cf_matrix
        MCC_numerator = (
                    current_cf_matrix[1, 1] * current_cf_matrix[0, 0] - current_cf_matrix[0, 1] * current_cf_matrix[1, 0])
        MCC_denominator = np.sqrt((current_cf_matrix[1, 1] + current_cf_matrix[0, 1]) * (
                    current_cf_matrix[1, 1] + current_cf_matrix[1, 0]) * (
                                              current_cf_matrix[0, 0] + current_cf_matrix[0, 1]) * (
                                              current_cf_matrix[0, 0] + current_cf_matrix[1, 0]))
        if MCC_denominator == 0:
            MCC_denominator = 1  # See wikipedia article for MCC - If denominator is zero can set to 1.
        MCCs[fold_counter_including_repeats] = MCC_numerator / MCC_denominator

        fold_no = fold_no + 1
        fold_counter_including_repeats = fold_counter_including_repeats + 1


cf_matrix = np.mean(cf_matrix_repeats, axis=2)
print('Standard deviation of means:',np.std(cf_matrix_repeats, axis=2)/np.sqrt(cf_matrix_repeats.shape[2]))
MCC_numerator = cf_matrix[1,1]*cf_matrix[0,0] - cf_matrix[0,1]*cf_matrix[1,0]
MCC_denominator = np.sqrt(  (cf_matrix[1,1] + cf_matrix[0,1]) * (cf_matrix[1,1] + cf_matrix[1,0])  * (cf_matrix[0,0] + cf_matrix[0,1]) * (cf_matrix[0,0] + cf_matrix[1,0]))
if MCC_denominator == 0:
    MCC_denominator = 1  # See wikipedia article for MCC - If denominator is zero can set to 1.
MCC = MCC_numerator / MCC_denominator
#print(MCC)

print(cf_matrix)
print(np.mean(MCCs), np.std(MCCs)/np.sqrt(len(MCCs)))

plt.plot(val_accuracy_summed)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

ax.set_title('CNN [64:32] Confusion Matrix [weighting=unit]\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()

################################################

