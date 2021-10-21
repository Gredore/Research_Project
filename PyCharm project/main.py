from cif_to_xyz_diffpy import run_cif_to_xyz_diffpy
from RDF import *
import numpy as np
from numba.typed import List

def main():

    mof_name = "ZIF-20"

    print("######### Converting .cif to .xyz")
    run_cif_to_xyz_diffpy(mof_name)

    print("######### Successfully converted .cif to .xyz")

    #Rs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    # Rs = [2]
    Rs = np.linspace(2, 30, num=40)

    # Makes a typed.List as lists are deprecating in Numba
    typed_Rs = List()
    [typed_Rs.append(x) for x in Rs]

    print("######### Calculating RDF")

    RDF = np.empty([len(typed_Rs), 2])
    length_one_unit_cell, xyz_array_float_stacked = rdf_setup("./"+mof_name+".xyz")
    RDF = rdf(RDF, length_one_unit_cell, xyz_array_float_stacked, typed_Rs, 10, 1, 1)

    print("######### Successfully calculated RDF")

    return RDF
#if __name__ == "__main__":

RDF = main()
