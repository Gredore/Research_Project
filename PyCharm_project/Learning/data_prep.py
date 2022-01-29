import numpy as np
from tensorflow.keras.utils import to_categorical

def data_prep():
    #Configure the preparation of the training set data
    RDF_Outputs_path = "../../RDF_Outputs/"
    r_spacing = 0.1
    largest_r = 60
    successful_s = [1,3,5,6,7,8,9,11,12,13,14,15,16,18,19,20]
    successful_u = [1, 2, 4, 5, 6, 7, 8, 9, 10]

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

    y_train_RDF = to_categorical(y_train_RDF)

    return X_train_RDF, y_train_RDF



