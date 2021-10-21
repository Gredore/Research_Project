import numpy as np
from numba import jit, prange
from time import time

def rdf_setup(xyz_file_path):

    # with open(xyz_file_path, "r") as xyz:
    #     num_xyz = xyz.readline()
    #     mof_name = xyz.readline()
    #
    #     xyz_array = np.array(xyz.readline().split(), ndmin=2)
    #     print("ok")
    #     for line in xyz:
    #         array_to_join = np.array(line.split(), ndmin=2) ##<== the slow step is np.array
    #         xyz_array = np.concatenate((xyz_array, array_to_join), axis=0)

    xyz_array = np.loadtxt("ZIF-20.xyz", dtype=str,skiprows=2)

    xyz_array_float = xyz_array[:, 1:4].astype(np.double)

    length_xyz_array_float = xyz_array_float.shape[0]
    length_one_unit_cell = int(length_xyz_array_float / 27)

    xyz_array_float_stacked = np.array(xyz_array_float[0:27, :], ndmin=3)
    xyz_array_float_stacked = np.swapaxes(xyz_array_float_stacked, 0, 1)

    for t in range(1, length_one_unit_cell):
        array_to_join = np.array(xyz_array_float[t*27:(t+1)*27, :], ndmin=3)
        array_to_join = np.swapaxes(array_to_join, 0, 1)
        xyz_array_float_stacked = np.concatenate((xyz_array_float_stacked, array_to_join), axis=1)

    return length_one_unit_cell, xyz_array_float_stacked



@jit(nopython=True, parallel=True)
def rdf_single_R(length_one_unit_cell, xyz_array_float_stacked, R, B, Pi, Pj):

    sum_RDF = 0

    for i in prange(0, length_one_unit_cell):
        for j in prange(0, length_one_unit_cell):
            if j > i:
                r = 1000
                for k in range(0, 27):
                #for k in range(13, 14):
                    euclid_dist = np.linalg.norm(xyz_array_float_stacked[13, i, 0:3] - xyz_array_float_stacked[k, j, 0:3])
                    if euclid_dist < r:
                        r = euclid_dist
                summand = Pi * Pj * np.exp(-B * ((r - R) ** 2))
                sum_RDF += summand

        # if i%100 == 0:
        #      print(i)
    return sum_RDF



@jit(nopython=True, parallel=True)
def rdf(RDF,length_one_unit_cell, xyz_array_float_stacked, Rs, B, Pi, Pj):
    for R_index in prange(0, len(Rs)):
        RDF_single_R=rdf_single_R(length_one_unit_cell, xyz_array_float_stacked, Rs[R_index], 10, 1, 1)
        RDF[R_index,0] = Rs[R_index]
        RDF[R_index,1] = RDF_single_R
        print(R_index)
    return RDF



