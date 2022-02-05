from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_prep import *

num_folds = 10
NCOMPONENTS = 80 #For Principal Component Analysis


#NOTE: Configuration MUST be done within data_prep.py before running!
X_train_RDF, y_train_RDF, class_weights_dict, num_stable_RDFs, num_unstable_RDFs = data_prep(use_catagorical_y=False)

#########################
# Code below used to plot all RDFs for interest

# fig, ax = plt.subplots()
# #Manually set r values
# ax.plot(np.linspace(0, 60, 600), X_train_RDF[:,:,0].T,linewidth=0.2)
# plt.show()

#########################

#########################
# Code below used to estimate number of components required to explain variance

# pca = PCA(n_components=len(X_train_RDF[:,:,0]))
# pca.fit(X_train_RDF[:,:,0])
#
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of components')
# plt.ylabel('Cumulative explained variance')
# plt.show()

############################



# Define per-fold score containers
acc_per_fold = []

kfold = KFold(n_splits=num_folds, shuffle=True)


fold_no = 1
for train, test in kfold.split(X_train_RDF, y_train_RDF):

    pca = PCA(n_components=NCOMPONENTS)
    X_pca_train = pca.fit_transform(X_train_RDF[train][:,:,0])
    X_pca_test = pca.transform(X_train_RDF[test][:,:,0])

    clf = make_pipeline(StandardScaler(), svm.SVC(C=1,kernel='rbf', class_weight='balanced'))
    #clf = svm.SVC(C=1,kernel='rbf', class_weight='balanced')  # clf is accepted shorthand for 'classifier'
    clf.fit(X_pca_train, y_train_RDF[train])

    y_pred = clf.predict(X_pca_test)
    print(y_train_RDF[test])
    print(y_pred)
    acc_per_fold.append(metrics.accuracy_score(y_train_RDF[test], y_pred)*100)
    #print("Accuracy:", metrics.accuracy_score(y_train_RDF[test], y_pred))

print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i + 1} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)/num_folds})')

print('------------------------------------------------------------------------')
print("Weighted random guessing would give accuracy: ", num_stable_RDFs/(num_stable_RDFs+num_unstable_RDFs))