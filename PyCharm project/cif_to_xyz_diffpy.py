from diffpy.structure import loadStructure
from diffpy.structure.expansion import supercell

def run_cif_to_xyz_diffpy(mof_name):
    structure_loaded = loadStructure(mof_name+'.cif')
    supercell_created = supercell(structure_loaded, [3, 3, 3])
    supercell_created.write(mof_name+'.xyz', 'xyz')
