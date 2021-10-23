from cif_to_xyz_diffpy import run_cif_to_xyz_diffpy
from RDF import *
import numpy as np
from numba.typed import List
import time

def main(mof_name, num_sample_rs):

    print("######### Converting .cif to .xyz")
    run_cif_to_xyz_diffpy(mof_name)

    print("######### Successfully converted .cif to .xyz")

    Rs = np.linspace(0, 30, num=num_sample_rs)

    # Makes a typed.List as lists are deprecating in Numba
    typed_Rs = List()
    [typed_Rs.append(x) for x in Rs]

    #s = time.time()

    print("######### RDF Calculations - Loading .xyz")
    xyz_array_float, length_one_unit_cell, atoms = rdf_load_xyz("./"+mof_name+".xyz")

    #Create empty stacked xyz
    xyz_array_float_stacked = np.zeros([27, length_one_unit_cell, 3])
    # Create empty unit cell atom list
    unit_cell_atoms = np.empty(shape=[length_one_unit_cell, 1])
    print("######### RDF Calculations - Stacking adjacent unit cells")
    xyz_array_float_stacked, unit_cell_atoms = rdf_setup(xyz_array_float, xyz_array_float_stacked, length_one_unit_cell, unit_cell_atoms, atoms)

    # Create empty RDF array
    RDF = np.zeros([len(typed_Rs), 2])

    print("######### RDF Calculations - Calculating RDF")
    RDF = rdf(RDF, length_one_unit_cell, xyz_array_float_stacked, typed_Rs, 10, 1, 1)

    #f = time.time()
    #print(f-s)
    print("######### Successfully calculated RDF")

    return RDF
#if __name__ == "__main__":

RDF = main("MIL-47", 300)
#Fourier_RDF = np.real(np.fft.rfft(RDF[:,1], axis=0))
