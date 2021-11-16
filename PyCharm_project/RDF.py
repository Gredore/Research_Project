import numpy as np
from numba import jit, prange, objmode
import math
import time
import sys


### These two functions are used to encode the element name into an int that can be parsed to numba functions (unlike str)
def convertStrToNumber(s):
    return int.from_bytes(s.encode(), 'little')

def convertStrFromNumber(n):
    return n.to_bytes(math.ceil(n.bit_length() / 8), 'little').decode()

def rdf_load_xyz(xyz_file_path):

    xyz_array = np.loadtxt(xyz_file_path, dtype=str,skiprows=2)
    xyz_array_float = xyz_array[:, 1:4].astype(float)

    length_xyz_array_float = xyz_array_float.shape[0]
    length_one_unit_cell = int(length_xyz_array_float / 27)

    atoms = np.array([convertStrToNumber(atom_i) for atom_i in xyz_array[:, 0]])
    atoms = np.transpose(atoms)
    # Convert strings to floats in the atoms array
    atoms = atoms.astype(np.float)

    all_element_property_vectors = np.loadtxt('Property_vectors.csv',dtype=str, delimiter=',', skiprows=1)
    all_element_property_vectors[:, 0] = np.transpose(np.array([convertStrToNumber(element_i) for element_i in all_element_property_vectors[:, 0]]))
    # Convert strings to floats in the element property vector array
    all_element_property_vectors = all_element_property_vectors.astype(np.float)

    return xyz_array_float, length_one_unit_cell, atoms, all_element_property_vectors

@jit(nopython=True, parallel=True)
def rdf_setup(xyz_array_float, xyz_array_float_stacked, length_one_unit_cell, unit_cell_atoms, atoms, all_element_property_vectors, all_unit_cell_property_vectors):

    for t in prange(0, length_one_unit_cell):
        array_to_join = xyz_array_float[t*27:(t+1)*27, :]
        xyz_array_float_stacked[:,t,:] = array_to_join
        atom_to_join = atoms[t*27]
        unit_cell_atoms[t] = atom_to_join
        all_element_property_vectors_atoms = all_element_property_vectors[:, 0]
        all_unit_cell_property_vectors[t] = all_element_property_vectors[np.where(all_element_property_vectors_atoms == atom_to_join)[0][0],1:np.shape(all_element_property_vectors)[1]]

    return xyz_array_float_stacked, unit_cell_atoms, all_unit_cell_property_vectors


@jit(nopython=True, parallel=True)
def rdf(RDF, length_one_unit_cell, xyz_array_float_stacked, Rs, B, property_vector, progress_proxy):
    #print(" -  RDF Calculations - 10 chunk complete messages expected:")
    for i in prange(0, length_one_unit_cell):

        for j in prange(0, length_one_unit_cell):
            if j > i:
                r = 1000
                for k in range(0, 27):
                    # for k in range(13, 14):
                    euclid_dist = np.linalg.norm(
                        xyz_array_float_stacked[13, i, 0:3] - xyz_array_float_stacked[k, j, 0:3])
                    if euclid_dist < r:
                        r = euclid_dist

                for R_index in range(0, len(Rs)):
                    R = Rs[R_index]
                    summand = property_vector[i] * property_vector[j] * np.exp(-B * ((r - R) ** 2))
                    RDF[R_index, 1] = RDF[R_index,1] + summand
                    RDF[R_index, 0] = R

        #if i % int(length_one_unit_cell / 10) == 0:
            #print(" -  RDF Calculations - Completed chunk (", i, ")")

        progress_proxy.update(1)
    return RDF

