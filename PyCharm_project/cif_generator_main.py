## Usage ##
# Run the this file. Make sure CCDC is installed and paths are set correctly in cif_generator_daughter.py and the path below.

import subprocess

CCDC_API_path = '/home/george/CCDC/Python_API_2021/'

process = subprocess.run([CCDC_API_path + 'run_csd_python_api', './cif_generator_daughter.py'], bufsize=0)
