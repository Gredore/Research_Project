import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
from random import sample
import random

def data_prep(use_catagorical_y, atomic_weighting, random_sample_ratio=None, random_seed=None):
    #Configure the preparation of the training set data
    RDF_Outputs_path = "../../RDF_Outputs_" + atomic_weighting + "/"
    r_spacing = 0.1
    largest_r = 60
    successful_s = [1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 99, 101, 102, 103, 104, 105, 106, 107, 108]
    successful_u = [1,2,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]

    if random_sample_ratio is not None:
        random.seed(random_seed)
        successful_s = sample(successful_s, round(len(successful_s)*random_sample_ratio))
        successful_u = sample(successful_u, round(len(successful_u)*random_sample_ratio))

    num_stable_RDFs = len(successful_s)
    num_unstable_RDFs = len(successful_u)


    Rs = np.arange(0, largest_r, r_spacing)

    X_train_RDF = np.zeros([num_stable_RDFs + num_unstable_RDFs, int(largest_r/r_spacing), 1], dtype=np.float32)
    y_train_RDF = np.zeros([num_stable_RDFs + num_unstable_RDFs])

    for i in range (0, num_stable_RDFs):
        s_index = successful_s[i]
        #RDF_path = RDF_Outputs_path + "s" + str(i+1) + ".csv"
        RDF_path = RDF_Outputs_path + "s" + str(s_index) + ".csv"

        csv_ingest = np.loadtxt(RDF_path,dtype=str, delimiter=',')
        rdf_values = csv_ingest[:,1]
        padded_rdf = np.pad(rdf_values, (0, int(largest_r/r_spacing) - rdf_values.size), 'constant')
        X_train_RDF[i, :, 0] = padded_rdf

        #Stable is defined as 1
        y_train_RDF[i] = 1

    for i in range (num_stable_RDFs, num_stable_RDFs + num_unstable_RDFs):
        u_index = successful_u[i-num_stable_RDFs]
        #RDF_path = RDF_Outputs_path + "s" + str(i+1) + ".csv"
        RDF_path = RDF_Outputs_path + "u" + str(u_index) + ".csv"

        csv_ingest = np.loadtxt(RDF_path,dtype=str, delimiter=',')
        rdf_values = csv_ingest[:,1]
        padded_rdf = np.pad(rdf_values, (0, int(largest_r/r_spacing) - rdf_values.size), 'constant')
        X_train_RDF[i, :, 0] = padded_rdf

        #Unstable is defined as 0
        y_train_RDF[i] = 0

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_RDF), y=y_train_RDF)
    class_weights_dict = dict(enumerate(class_weights))
    if use_catagorical_y:
        y_train_RDF = to_categorical(y_train_RDF)

    return X_train_RDF, y_train_RDF, class_weights_dict, num_stable_RDFs, num_unstable_RDFs



