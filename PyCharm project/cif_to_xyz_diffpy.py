from diffpy.Structure import loadStructure
from diffpy.Structure.expansion import supercell

mof_name = "IRMOF-1"
structure_loaded = loadStructure(mof_name+'.cif')
supercell_created = supercell(structure_loaded, [3, 3, 3])
supercell_created.write(mof_name+'.xyz', 'xyz')