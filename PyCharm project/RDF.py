import numpy as np
from numba import jit, prange, objmode
import time
import sys

def rdf_load_xyz(xyz_file_path):

    xyz_array = np.loadtxt(xyz_file_path, dtype=str,skiprows=2)
    xyz_array_float = xyz_array[:, 1:4].astype(float)

    length_xyz_array_float = xyz_array_float.shape[0]
    length_one_unit_cell = int(length_xyz_array_float / 27)

    atoms = np.array([ord(atom_i) for atom_i in xyz_array[:, 0]])
    atoms = np.transpose(atoms)

    return xyz_array_float, length_one_unit_cell, atoms

@jit(nopython=True, parallel=True)
def rdf_setup(xyz_array_float, xyz_array_float_stacked, length_one_unit_cell, unit_cell_atoms, atoms):

    for t in prange(0, length_one_unit_cell):
        array_to_join = xyz_array_float[t*27:(t+1)*27, :]
        xyz_array_float_stacked[:,t,:] = array_to_join
        atom_to_join = atoms[t*27]
        unit_cell_atoms[t] = atom_to_join

    return xyz_array_float_stacked, unit_cell_atoms

@jit(nopython=True, parallel=True)
def rdf(RDF, length_one_unit_cell, xyz_array_float_stacked, Rs, B, Pi, Pj):
    print("######### RDF Calculations - 10 chunk complete messages expected:")
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
                    summand = Pi * Pj * np.exp(-B * ((r - R) ** 2))
                    RDF[R_index, 1] = RDF[R_index,1] + summand
                    RDF[R_index, 0] = R

        if i % int(length_one_unit_cell / 10) == 0:
            print("######### RDF Calculations - Completed chunk (", i, ")")

    return RDF

