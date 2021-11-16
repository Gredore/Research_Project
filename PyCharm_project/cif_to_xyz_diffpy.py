from diffpy.structure import loadStructure
from diffpy.structure.expansion import supercell

def run_cif_to_xyz_diffpy(mof_name, cif_path, xyz_path):
    structure_loaded = loadStructure(cif_path + mof_name+'.cif')
    supercell_created = supercell(structure_loaded, [3, 3, 3])
    supercell_created.write(xyz_path+mof_name+'.xyz', 'xyz')

