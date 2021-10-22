from cif_to_xyz_diffpy import run_cif_to_xyz_diffpy
from RDF import *
import numpy as np
from numba.typed import List
import time

def main():

    mof_name = "ZIF-20"

    print("######### Converting .cif to .xyz")
    run_cif_to_xyz_diffpy(mof_name)

    print("######### Successfully converted .cif to .xyz")

    #Rs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 265, 266]
    # Rs = [2]
    Rs = np.linspace(2, 30, num=100)
    #Rs = np.array([265, 266])

    # Makes a typed.List as lists are deprecating in Numba
    typed_Rs = List()
    [typed_Rs.append(x) for x in Rs]

    #s = time.time()

    print("######### RDF Calculations - Loading .xyz")
    xyz_array_float, length_one_unit_cell = rdf_load_xyz("./"+mof_name+".xyz")

    #Create empty stacked xyz
    xyz_array_float_stacked = np.zeros([27, length_one_unit_cell, 3])

    print("######### RDF Calculations - Stacking adjacent unit cells")
    xyz_array_float_stacked = rdf_setup2(xyz_array_float, xyz_array_float_stacked, length_one_unit_cell)

    # Create empty RDF array
    RDF = np.zeros([len(typed_Rs), 2])

    print("######### RDF Calculations - Calculating RDF")
    RDF = rdf(RDF, length_one_unit_cell, xyz_array_float_stacked, typed_Rs, 10, 1, 1)

    #f = time.time()
    #print(f-s)
    print("######### Successfully calculated RDF")

    return RDF
#if __name__ == "__main__":

RDF = main()
