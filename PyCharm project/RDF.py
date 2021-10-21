import numpy as np
from numba import jit, prange
from numba.typed import List

def rdf_setup(xyz_file_path):

    with open(xyz_file_path, "r") as xyz:
        num_xyz = xyz.readline()
        mof_name = xyz.readline()

        xyz_array = np.array(xyz.readline().split(), ndmin=2)

        for line in xyz:
            array_to_join = np.array(line.split(), ndmin=2)
            xyz_array = np.concatenate((xyz_array, array_to_join), axis=0)

    xyz_array_float = xyz_array[:, 1:4].astype(np.double)

    length_xyz_array_float = xyz_array_float.shape[0]
    length_one_unit_cell = int(length_xyz_array_float / 27)

    xyz_array_float_stacked = np.array(xyz_array_float[0:length_one_unit_cell, :], ndmin=3)
    for t in range(1, 27):
        array_to_join = np.array(xyz_array_float[t*length_one_unit_cell:(t+1)*length_one_unit_cell, :], ndmin=3)
        xyz_array_float_stacked = np.concatenate((xyz_array_float_stacked, array_to_join), axis=0)

    return length_one_unit_cell, xyz_array_float_stacked

@jit(nopython=True, parallel=True)
def rdf_single_R(length_one_unit_cell, xyz_array_float_stacked, R, B, Pi, Pj):

    sum_RDF = 0

    for i in prange(0, length_one_unit_cell):
        for j in range(0, length_one_unit_cell):
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
        #     print(i)
    return sum_RDF

@jit(nopython=True, parallel=True)
def rdf(RDF,length_one_unit_cell, xyz_array_float_stacked, Rs, B, Pi, Pj):

    for R_index in prange(0, len(Rs)):
        RDF_single_R=rdf_single_R(length_one_unit_cell, xyz_array_float_stacked, Rs[R_index], 10, 1, 1)
        RDF[R_index,0] = Rs[R_index]
        RDF[R_index,1] = RDF_single_R
        print(R_index)
    return RDF

Rs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]


#Makes a typed.List as lists are deprecating in Numba
typed_Rs = List()
[typed_Rs.append(x) for x in Rs]

RDF = np.empty([len(typed_Rs), 2])
length_one_unit_cell, xyz_array_float_stacked = rdf_setup("./.RASPA_Output/zns222.xyz")
RDF = rdf(RDF,length_one_unit_cell, xyz_array_float_stacked, typed_Rs, 10, 1, 1)


