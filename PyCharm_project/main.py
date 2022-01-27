from cif_to_xyz_diffpy import run_cif_to_xyz_diffpy
from RDF import *
import numpy as np
from numba.typed import List
import matplotlib.pyplot as plt
import csv
import sys
from numba_progress import ProgressBar
import time
import traceback

def delete_last_lines(n):
    for _ in range(n):
        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[2K')

def main_rdf(mof_name, cif_path, xyz_path, num_sample_rs, largest_r, B, property_index):

    s = time.time()

    print(" -  Converting .cif to .xyz")
    run_cif_to_xyz_diffpy(mof_name, cif_path, xyz_path)

    f = time.time()

    #If .xyz took too long to make, then skip MOF
    if f-s > 10:
        print("#### Time to create .xyz implies file is too large... Skipping " + mof_name)
        return

    print(" -  Successfully converted .cif to .xyz")

    Rs = np.linspace(0, largest_r, num=num_sample_rs)

    # Makes a typed.List as lists are deprecating in Numba
    typed_Rs = List()
    [typed_Rs.append(x) for x in Rs]


    print(" -  RDF Calculations - Loading .xyz")
    xyz_array_float, length_one_unit_cell, atoms, all_element_property_vectors = rdf_load_xyz(xyz_path + mof_name+".xyz")

    # Create empty stacked xyz
    xyz_array_float_stacked = np.zeros([27, length_one_unit_cell, 3])
    # Create empty unit cell atom list
    unit_cell_atoms = np.empty(shape=[length_one_unit_cell, 1])
    # Create empty unit cell property vector
    all_unit_cell_property_vectors = np.zeros(shape=[length_one_unit_cell,np.shape(all_element_property_vectors)[1]-1])
    print(" -  RDF Calculations - Stacking adjacent unit cells")
    xyz_array_float_stacked, unit_cell_atoms, all_unit_cell_property_vectors = rdf_setup(xyz_array_float, xyz_array_float_stacked, length_one_unit_cell, unit_cell_atoms, atoms, all_element_property_vectors, all_unit_cell_property_vectors)

    # If an element in the .xyz is not in the property vector then need to abondon and log it.
    # 'None' is returned by rdf_setup in this case
    if xyz_array_float_stacked is None:
        raise ValueError('Element not found in Property_vectors.csv - Abandoning')

    # Create empty RDF array
    RDF = np.zeros([len(typed_Rs), 2])

    print(" -  RDF Calculations - Calculating RDF")
    with ProgressBar(total=length_one_unit_cell) as progress:
        RDF = rdf(RDF, length_one_unit_cell, xyz_array_float_stacked, typed_Rs, B, all_unit_cell_property_vectors[:,property_index], progress)

    RDF_scaled = RDF.copy()

    #Old scaling system scaled from -1 to 1.
    #RDF_scaled[:, 1] = -1 + ((RDF[:, 1] - np.min(RDF[:, 1])) * 2 / (np.max(RDF[:, 1]) - np.min(RDF[:, 1])))

    area_under_curve = np.trapz(RDF[:,1], RDF[:,0])

    #Scales RDF such that probability = 1 for all positions.
    RDF_scaled[:,1] = RDF[:,1] / area_under_curve

    print(" -  Successfully calculated RDF")

    return RDF_scaled

cif_path = '../CIF_Outputs/'
xyz_path = '../XYZ_Outputs/'
rdf_path = '../RDF_Outputs/'

name_list = []
failed_list = []
with open('input_names_reduced.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        name_list.append(row[0])

#name_list = [name_list[0]]
for iteration, name in enumerate(name_list):
    print('\033[0m\033[1m'+"Current MOF: " + name + " (" + str(iteration+1) + " / " + str(len(name_list)) +  ")" + '\033[0m \033[3m\033[2m')

    #try clause used to ensure script can be left running and problem cases flagged for later investigation.
    try:
        ##            main_rdf(mof_name, cif_path, xyz_path, num_sample_rs, largest_r, B, property_index)
        RDF_scaled0 = main_rdf(name, cif_path, xyz_path, 300, 30, 200, 0)


        #None is returned if .xyz took too long to make
        if RDF_scaled0 is not None:
            np.savetxt(rdf_path + name + '.csv', RDF_scaled0, delimiter=',')
            #delete_last_lines(19)
            print('\033[0m\033[1m' + "Completed MOF: " + name + " (" + str(iteration+1) + " / " + str(len(name_list)) + ")")
        else:
            failed_list.append(name)

    except Exception as e:
        print(traceback.format_exc())
        failed_list.append(name)

print('\033[0m\033[1m' + "Finished.")
if len(failed_list) > 0:
    print("This is the list of failed entries", failed_list)
# RDF_scaled1 = main("IRMOF-1", 300, 1)
# RDF_scaled3 = main("IRMOF-1", 300, 3)
#Fourier_RDF = np.real(np.fft.rfft(RDF_scaled[:,1], axis=0))
#plt.plot(RDF_scaled0[:,0],RDF_scaled0[:,1])
# plt.plot(RDF_scaled0[:,0],RDF_scaled0[:,1]-RDF_scaled1[:,1])
# plt.plot(RDF_scaled0[:,0],RDF_scaled0[:,1]-RDF_scaled3[:,1])
#plt.show()

##NOTE: RUNNING main.py FROM COMMAND LINE SEEMS BETTER AT FORCING ALL CORES TO BE USED##
