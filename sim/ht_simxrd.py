from Pysimxrd import generator
from ase.db import connect
from ase import Atoms
import random
import concurrent.futures
from tqdm import tqdm

def process_entry(_id, database, times):
    """
    Processes a single entry from the database to simulate X-ray diffraction patterns.
    
    Parameters:
        _id (int): ID of the entry in the database.
        database (ase.db.Database): The database object to retrieve atomic structures.
        times (int): Number of times to simulate patterns for the entry.

    Returns:
        tuple: A tuple containing the following elements:
            - atom_list (list): List of atomic structures.
            - _target (list): List of labels for each simulated pattern.
            - _chem_form (list): List of chemical formulas for the simulated patterns.
            - _latt_dis (list): List of lattice distances.
            - _inten (list): List of simulated intensities.
    """
    atom_list = []
    _target = []
    _chem_form = []
    _latt_dis = []
    _inten = []
 

    try:
        entry = database.get(id=_id)
        # label = entry['Label']
        # Pay attention: The MP database is labeled first and saved under the key 'Label' in our data.db.
        # Here, we assign a label as the data ID. We welcome sharing processed MP data with groups that wish to collaborate with us.
        # Please contact Mr. Cao at bcao686@connect.hkust-gz.edu.cn.

        label = _id + 1
        atoms = entry.toatoms()

        for sim in range(times):
            atom_list.append(atoms)
            _target.append(label)
            _chem_form.append(atoms.get_chemical_formula())

            # Parameters for the parser function
            """
            Simulate X-ray diffraction patterns based on a given database file and data ID.
        
            Parameters:
                db_file (str): Path to the database file (e.g., 'cif.db').
                data_id (int): The ID of the data entry to be processed.
        
            Optional Parameters:
                deformation (bool, optional): Whether to apply deformation to the lattice. Defaults to False.
                sim_model (str, optional): The simulation model to use. Can be 'WPEM' for WPEM simulation or None for conventional simulation. Defaults to None.
                xrd (str, optional): The type of X-ray diffraction to simulate. Can be 'reciprocal' or 'real'. Defaults to 'reciprocal'.
                
                Sample Parameters:
                grainsize (float, optional): Grain size of the specimen in Angstroms. Defaults to 20.0.
                perfect_orientation (list of float, optional): Perfect orientation of the specimen in degrees. Defaults to [0.1, 0.1].
                lattice_extinction_ratio (float, optional): Ratio of lattice extinction in deformation. Defaults to 0.01.
                lattice_torsion_ratio (float, optional): Ratio of lattice torsion in deformation. Defaults to 0.01.
                
                Testing Condition Parameters:
                thermo_vibration (float, optional): Thermodynamic vibration, the average offset of atoms, in Angstroms. Defaults to 0.1.
                background_order (int, optional): The order of the background. Can be 4 or 6. Defaults to 6.
                background_ratio (float, optional): Ratio of scattering background intensity to peak intensity. Defaults to 0.05.
                mixture_noise_ratio (float, optional): Ratio of mixture vibration noise to peak intensity. Defaults to 0.02.
                
                Instrument Parameters:
                dis_detector2sample (int, optional): Distance between the detector and the sample in mm. Defaults to 500.
                half_height_slit_detector (int, optional): Half height of the slit-shaped detector in mm. Defaults to 5 (2H = 10 mm).
                half_height_sample (int, optional): Half height of the sample in mm. Defaults to 2.5 (height = 5 mm).
                zero_shift (float, optional): Zero shift of angular position in degrees. Defaults to 0.1.
        
            Returns:
                tuple: A tuple containing the following elements:
                    - x: Lattice plane distance in the x-direction (in Angstroms) if xrd='real', or diffraction angle in the x-direction (in degrees) if xrd='reciprocal'.
                    - y: Corresponding diffraction intensity in the y-direction (arbitrary units).
            """
            deformation = True
            grainsize = random.uniform(2, 20)
            orientation = [random.uniform(0., 0.4), random.uniform(0., 0.4)]
            thermo_vib = random.uniform(0.0, 0.3)
            lattice_extinction_ratio = 0.01
            lattice_torsion_ratio = 0.01
            background_order = 6
            background_ratio = 0.05
            mixture_noise_ratio = 0.02
            dis_detector2sample = 500
            half_height_slit_detector_H = 50
            half_height_sample_S = 25
            zero_shift = random.uniform(-1.5, 1.5)

            # Generate the simulated diffraction pattern
            x, y = generator.parser(
                database=database, entry_id=_id,
                deformation=deformation, grainsize=grainsize, prefect_orientation=orientation,
                thermo_vibration=thermo_vib, lattice_extinction_ratio=lattice_extinction_ratio,
                lattice_torsion_ratio=lattice_torsion_ratio, background_order=background_order,
                background_ratio=background_ratio, mixture_noise_ratio=mixture_noise_ratio,
                dis_detector2sample=dis_detector2sample, half_height_slit_detector=half_height_slit_detector_H,
                half_height_sample =half_height_sample_S, zero_shift=zero_shift
            )

            _latt_dis.append(str(x))
            _inten.append(str(y))

    except Exception as e:
        print(f"An error occurred: crystal id = {_id}, error: {e}")
        return None

    return  atom_list, _target, _chem_form, _latt_dis, _inten


def simulator(db_file, sv_file, times=10):
    """
    Simulates X-ray diffraction patterns for entries in a database and saves the results.

    Parameters:
        db_file (str): Path to the input database file containing crystal structures.
        sv_file (str): Path to save the output database containing simulated patterns.
        times (int): Number of simulated patterns to generate for each entry.

    Returns:
        bool: True if the simulation completes successfully.
    """
    database = connect(db_file)
    total_entries = database.count()

    entries = list(range(1, total_entries + 1))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(process_entry, _id, database, times) for _id in entries]

        atom_lists, target_lists, chem_form_lists, latt_dis_lists, inten_lists = [], [], [], [],[]
        for future in tqdm(concurrent.futures.as_completed(results), total=len(entries), desc='Processing Entries'):
            result = future.result()
            if result is not None:
                atom_list, target, chem_form, latt_dis, inten = result
                atom_lists.extend(atom_list)
                target_lists.extend(target)
                chem_form_lists.extend(chem_form)
                latt_dis_lists.extend(latt_dis)
                inten_lists.extend(inten)

    databs = connect(sv_file)
    for k in tqdm(range(len(target_lists)), desc='Writing to Database'):
        id = k  # Pay attention
        try:
            atoms = Atoms(atom_lists[k])
            databs.write(atoms=atoms, latt_dis=latt_dis_lists[k], intensity=inten_lists[k], Label=target_lists[k])
        except Exception as e:
            print("An error occurred while writing to the database: ", e)
    return True


if __name__ == '__main__':
    db_file = './demo_mp.db'
    sv_file = './train.db'
    simulator(db_file, sv_file, times=1)
