import csv
import os

xyz_file_path = "./.RASPA_Output/ZIF-68.xyz"

f = open(xyz_file_path, "r")

with open(xyz_file_path, "r") as xyz, open('./.RASPA_Output/temp_csv.csv', 'w+', newline='') as csvfile:
    num_xyz = xyz.readline()
    output_name = xyz.readline()

    output_csv = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for line in xyz:
        output_csv.writerow(line.split())

    os.rename("./.RASPA_Output/temp_csv.csv", "./.RASPA_Output/"+output_name+".csv")
