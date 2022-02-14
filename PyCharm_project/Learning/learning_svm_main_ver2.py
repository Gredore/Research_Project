import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.utils.fixes import loguniform
from sklearn.metrics import matthews_corrcoef
import sklearn.metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
import seaborn as sns

from data_prep import *

X_train_RDF_unit, y_train_RDF, class_weights_dict, num_stable_RDFs, num_unstable_RDFs = data_prep(use_catagorical_y=False, atomic_weighting="unit")
X_train_RDF_electroneg, _, _, _, _ = data_prep(use_catagorical_y=False, atomic_weighting="electroneg")

X_train_RDF_unit_minus_electroneg = X_train_RDF_unit - X_train_RDF_electroneg

#X_train_RDF = np.concatenate((X_train_RDF_unit, X_train_RDF_electroneg, X_train_RDF_unit_minus_electroneg), axis=1)

X_train_RDF_unit_fft = np.abs(np.fft.rfft(X_train_RDF_unit, axis=1))
X_train_RDF_unit_minus_electroneg_fft = np.abs(np.fft.rfft(X_train_RDF_unit_minus_electroneg, axis=1))

#X_train_RDF = np.concatenate((X_train_RDF_unit, X_train_RDF_electroneg, X_train_RDF_unit_minus_electroneg, X_train_RDF_unit_fft, X_train_RDF_unit_minus_electroneg_fft), axis=1)
X_train_RDF = X_train_RDF_electroneg

#########################
#Code below used to plot all RDFs for interest

# fig, ax = plt.subplots()
# #Manually set r values
# plt.rcParams.update({'font.size': 16})
# ax.plot(np.linspace(0, 60, 300), X_train_RDF[:,0:300,0].T,linewidth=0.8)
# plt.xlabel('R / $\AA$')
# plt.ylabel('RDF score')
# plt.show()

#########################


clf = make_pipeline(StandardScaler(), svm.SVC(kernel='rbf', class_weight='balanced'))

param_distributions = {
    'svc__C': loguniform(1e-2, 1e2),
}

#print(clf.get_params().keys())

num_folds = 15

if True: #Turn off and on RandomizedSearch for C
    matthews_corrcoef_scorer = make_scorer(sklearn.metrics.matthews_corrcoef)
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
    gd_sr = RandomizedSearchCV(estimator=clf,
                         param_distributions=param_distributions,
                         scoring=matthews_corrcoef_scorer,
                         cv=kfold,
                         n_iter=400,
                         n_jobs=-1)

    gd_sr.fit(X_train_RDF[:,:,0], y_train_RDF)

    best_parameters = gd_sr.best_params_
    print(best_parameters)
    best_result = gd_sr.best_score_
    print(best_result)

    #print(gd_sr.cv_results_['mean_test_score'])
    #print(gd_sr.cv_results_['std_test_score'])


    ################################################
    #Create figure to show different values of C and the MCC score for each:
    fig, ax = plt.subplots()
    ax.plot(gd_sr.cv_results_['param_svc__C'], gd_sr.cv_results_['mean_test_score'], 'k+')
    plt.xscale("log")
    plt.show()

    ################################################
    CSV_C = best_parameters['svc__C']
else:
    CSV_C = 4

#Generate averaged confusion matrix
Repeats_of_shuffled_splits = 20

cf_matrix_repeats = np.zeros([2, 2, Repeats_of_shuffled_splits*num_folds])
MCCs = np.zeros([Repeats_of_shuffled_splits*num_folds])

np.set_printoptions(threshold=np.inf)

fold_counter_including_repeats = 0
for i in range (0, Repeats_of_shuffled_splits):
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
    for train, test in kfold.split(X_train_RDF, y_train_RDF):

        X_train = X_train_RDF[train][:, :, 0]
        X_test = X_train_RDF[test][:, :, 0]

        clf = make_pipeline(StandardScaler(), svm.SVC(C=CSV_C, kernel='rbf', class_weight='balanced'))
        clf.fit(X_train, y_train_RDF[train])

        #Occasionally fails due to some kind of scaling issue - Just re-run to fix.
        y_pred = clf.predict(X_test)

        current_cf_matrix = confusion_matrix(y_train_RDF[test], y_pred)
        current_cf_matrix = current_cf_matrix/(current_cf_matrix[1,1])
        cf_matrix_repeats[:,:,fold_counter_including_repeats] = current_cf_matrix
        MCC_numerator = (current_cf_matrix[1,1]*current_cf_matrix[0,0] - current_cf_matrix[0,1]*current_cf_matrix[1,0])
        MCC_denominator = np.sqrt(  (current_cf_matrix[1,1] + current_cf_matrix[0,1]) * (current_cf_matrix[1,1] + current_cf_matrix[1,0])  * (current_cf_matrix[0,0] + current_cf_matrix[0,1]) * (current_cf_matrix[0,0] + current_cf_matrix[1,0]))
        if MCC_denominator == 0:
            MCC_denominator = 1 #See wikipedia article for MCC - If denominator is zero can set to 1.
        MCCs[fold_counter_including_repeats] = MCC_numerator/MCC_denominator
        #print(y_train_RDF[test])
        #print(confusion_matrix(y_train_RDF[test], y_pred))
        fold_counter_including_repeats = fold_counter_including_repeats + 1

cf_matrix = np.mean(cf_matrix_repeats, axis=2)
print('Standard deviation of means:',np.std(cf_matrix_repeats, axis=2)/np.sqrt(cf_matrix_repeats.shape[2]))
MCC_numerator = cf_matrix[1,1]*cf_matrix[0,0] - cf_matrix[0,1]*cf_matrix[1,0]
MCC_denominator = np.sqrt(  (cf_matrix[1,1] + cf_matrix[0,1]) * (cf_matrix[1,1] + cf_matrix[1,0])  * (cf_matrix[0,0] + cf_matrix[0,1]) * (cf_matrix[0,0] + cf_matrix[1,0]))
if MCC_denominator == 0:
    MCC_denominator = 1  # See wikipedia article for MCC - If denominator is zero can set to 1.
MCC = MCC_numerator / MCC_denominator

print(cf_matrix)
#print(MCC)
print(np.mean(MCCs), np.std(MCCs)/np.sqrt(len(MCCs)))

ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

ax.set_title('SVM Confusion Matrix [weighting=electronegativity]\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()

################################################