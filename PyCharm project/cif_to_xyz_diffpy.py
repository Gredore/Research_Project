from diffpy.Structure import loadStructure
from diffpy.Structure.expansion import supercell
zns = loadStructure('IRMOF-1.cif')
zns222 = supercell(zns, [3, 3, 3])
zns222.write('zns222.xyz', 'xyz')