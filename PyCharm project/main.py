import subprocess

from utils.preprocess import write_raspa_input_file
from utils.preprocess import raspa_create_cif


def main():


    ###############################################################

    input_files_directory = "../RASPA_Output"
    output_files_directory = "../Outputs"
    path_to_raspa = "./Raspa_simulate"

    mof_name = "graphite"

    ##### At some point, make the above into separate file ########


    print("==================RUNNING RASPA==================")

    sim_input_file_path = write_raspa_input_file(
        simulation_type="MonteCarlo",
        number_of_cycles=0,
        print_every=1,
        framework=0,
        framework_name=mof_name,
        unit_cells=[1, 1, 1]
    )


    ##Run RASPA to obtain supercell

    print([path_to_raspa, sim_input_file_path])
    subprocess.run([path_to_raspa, sim_input_file_path])

    cif_dest_path, xyz_dest_path = raspa_create_cif(mof_name)

    print("==================RUNNING OBABEL==================")

    ##Run OBabel to obtain xyz of supercell obtained by RASPA
    subprocess.run(["obabel",  "-icif", cif_dest_path, "-oxyz", "-O", xyz_dest_path])


if __name__ == "__main__":
    main()