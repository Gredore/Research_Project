from sklearn.model_selection import RandomizedSearchCV
from sklearn import svm
from sklearn import metrics
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

X_train_RDF, y_train_RDF, class_weights_dict, num_stable_RDFs, num_unstable_RDFs = data_prep(use_catagorical_y=False)

num_folds = num_stable_RDFs + num_unstable_RDFs

clf = make_pipeline(StandardScaler(), svm.SVC(kernel='rbf', class_weight='balanced'))

param_distributions = {
    'svc__C': loguniform(1e-4, 1e5)
}

#print(clf.get_params().keys())

matthews_corrcoef_scorer = make_scorer(sklearn.metrics.matthews_corrcoef)
gd_sr = RandomizedSearchCV(estimator=clf,
                     param_distributions=param_distributions,
                     scoring=matthews_corrcoef_scorer,
                     cv=10,
                     n_iter=100,
                     n_jobs=-1)

gd_sr.fit(X_train_RDF[:,:,0], y_train_RDF)

best_parameters = gd_sr.best_params_
print(best_parameters)
best_result = gd_sr.best_score_
print(best_result)

#print(gd_sr.cv_results_['mean_test_score'])
#print(gd_sr.cv_results_['std_test_score'])

fig, ax = plt.subplots()
ax.plot(gd_sr.cv_results_['param_svc__C'], gd_sr.cv_results_['mean_test_score'], 'k+')
plt.xscale("log")
plt.show()

cf_matrix = confusion_matrix(y_train_RDF, gd_sr.predict(X_train_RDF[:,:,0]))
print(sklearn.metrics.matthews_corrcoef(y_train_RDF, gd_sr.predict(X_train_RDF[:,:,0])))

ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()