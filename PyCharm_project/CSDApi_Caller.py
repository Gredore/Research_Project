from ccdc import io
csd_reader = io.EntryReader('CSD')
first_csd_entry = csd_reader[0]

mol = first_csd_entry.crystal.molecule
print(mol.identifier)