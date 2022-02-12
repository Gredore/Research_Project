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

X_train_RDF_unit_minus_electroneg_fft = np.abs(np.fft.rfft(X_train_RDF_unit_minus_electroneg, axis=1))

X_train_RDF = np.concatenate((X_train_RDF_unit, X_train_RDF_electroneg, X_train_RDF_unit_minus_electroneg, X_train_RDF_unit_minus_electroneg_fft), axis=1)

#########################
# Code below used to plot all RDFs for interest

# fig, ax = plt.subplots()
# #Manually set r values
# ax.plot(np.linspace(0, 60, 301), X_train_RDF[:,:,0].T,linewidth=0.2)
# plt.show()

#########################


clf = make_pipeline(StandardScaler(), svm.SVC(kernel='rbf', class_weight='balanced'))

param_distributions = {
    'svc__C': loguniform(1e-4, 1e5)
}

#print(clf.get_params().keys())

num_folds = 10

if True: #Turn off and on RandomizedSearch for C
    matthews_corrcoef_scorer = make_scorer(sklearn.metrics.matthews_corrcoef)
    gd_sr = RandomizedSearchCV(estimator=clf,
                         param_distributions=param_distributions,
                         scoring=matthews_corrcoef_scorer,
                         cv=num_folds,
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
    CSV_C = 4.89

#Generate averaged confusion matrix
Repeats_of_shuffled_splits = 20

cf_matrix_repeats = np.zeros([2, 2, Repeats_of_shuffled_splits])

np.set_printoptions(threshold=np.inf)

for i in range (0, Repeats_of_shuffled_splits):
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
    fold_no = 1
    for train, test in kfold.split(X_train_RDF, y_train_RDF):
        X_train = X_train_RDF[train][:, :, 0]
        X_test = X_train_RDF[test][:, :, 0]

        clf = make_pipeline(StandardScaler(), svm.SVC(C=CSV_C, kernel='rbf', class_weight='balanced'))
        #clf = make_pipeline(svm.SVC(C=CSV_C, kernel='rbf', class_weight='balanced'))
        clf.fit(X_train, y_train_RDF[train])

        #Occasionally fails due to some kind of scaling issue - Just re-run to fix.
        y_pred = clf.predict(X_test)

        cf_matrix_repeats[:,:,i] = confusion_matrix(y_train_RDF[test], y_pred)


cf_matrix = np.mean(cf_matrix_repeats, axis=2)
print(np.std(cf_matrix_repeats, axis=2))
print('Standard deviation of means:',np.std(cf_matrix_repeats, axis=2)/np.sqrt(Repeats_of_shuffled_splits))

ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()

################################################