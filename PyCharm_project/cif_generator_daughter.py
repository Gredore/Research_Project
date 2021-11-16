## Usage ##
# Run the cif_generator_main.py file. Make sure CCDC is installed and paths are set correctly in cif_generator_main.py and the paths below.

output_folder_path = '/home/george/Documents/Research_Project/CIF_Outputs/'
input_names_path = '/home/george/Documents/Research_Project/PyCharm_project/input_names.csv'

from ccdc.search import TextNumericSearch
from ccdc import io
import csv
import os

name_list = []

with open(input_names_path) as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		name_list.append(row[0])

csd_reader = io.CrystalReader('CSD')

#print(name_list)

for name in name_list:
	try:
		crystal  = csd_reader.crystal(name)
		io.CrystalWriter(os.path.join(output_folder_path + name + '.cif')).write(crystal)
	except:
		print(name + ' has failed.')



