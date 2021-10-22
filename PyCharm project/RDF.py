import numpy as np
from numba import jit, prange, objmode
import time


def rdf_load_xyz(xyz_file_path):

    xyz_array = np.loadtxt(xyz_file_path, dtype=str,skiprows=2)
    xyz_array_float = xyz_array[:, 1:4].astype(float)

    length_xyz_array_float = xyz_array_float.shape[0]
    length_one_unit_cell = int(length_xyz_array_float / 27)

    return xyz_array_float, length_one_unit_cell

#@jit(nopython=True, parallel=True)
def rdf_setup(xyz_array_float, length_one_unit_cell):

    # with open(xyz_file_path, "r") as xyz:
    #     num_xyz = xyz.readline()
    #     mof_name = xyz.readline()
    #
    #     xyz_array = np.array(xyz.readline().split(), ndmin=2)
    #     print("ok")
    #     for line in xyz:
    #         array_to_join = np.array(line.split(), ndmin=2) ##<== the slow step is np.array
    #         xyz_array = np.concatenate((xyz_array, array_to_join), axis=0)

    xyz_array_float_stacked = np.array(xyz_array_float[0:27, :], ndmin=3)
    xyz_array_float_stacked = np.swapaxes(xyz_array_float_stacked, 0, 1)

    for t in range(1, length_one_unit_cell):
        array_to_join = np.array(xyz_array_float[t*27:(t+1)*27, :], ndmin=3)
        array_to_join = np.swapaxes(array_to_join, 0, 1)
        xyz_array_float_stacked = np.concatenate((xyz_array_float_stacked, array_to_join), axis=1)

    return length_one_unit_cell, xyz_array_float_stacked

@jit(nopython=True, parallel=True)
def rdf_setup2(xyz_array_float, xyz_array_float_stacked, length_one_unit_cell):

    for t in prange(0, length_one_unit_cell):
        array_to_join = xyz_array_float[t*27:(t+1)*27, :]
        xyz_array_float_stacked[:,t,:] = array_to_join

    return xyz_array_float_stacked

@jit(nopython=True, parallel=True)
def rdf(RDF, length_one_unit_cell, xyz_array_float_stacked, Rs, B, Pi, Pj):
    with objmode(start_time='f8'):
        start_time = time.perf_counter()
    printer = False

    for i in prange(0, length_one_unit_cell):

        if i % int(length_one_unit_cell / 10) == 0:
            with objmode(inner_start_time='f8'):
                inner_start_time = time.perf_counter()
            printer = True

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

        if printer:
            printer = False
            with objmode():
                inner_end_time = time.perf_counter()
                total_time_estimate = (inner_end_time-inner_start_time)*length_one_unit_cell
                total_time_passed = inner_end_time-start_time
                print("Estimated total time: ", total_time_estimate, "Time passed:", total_time_passed)



    return RDF

