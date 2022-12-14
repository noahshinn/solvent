import os
import re
import sys
import json
import warnings

from typing import List, NamedTuple

assert len(sys.argv) == 5
REACTANT_DIR = sys.argv[1]
PRODUCT_DIR = sys.argv[2]
WRITE_FILE = sys.argv[3]
NATOMS = int(sys.argv[4])


class NACData(NamedTuple):
    atom_types: List[List[str]]
    coords: List[List[List[float]]]
    energy_differences: List[float]
    norms: List[float]
    nacs: List[List[List[float]]]

def extract_data_from_meci_logs(reactant_dir: str, product_dir: str, write_file: str, natoms: int) -> None:
    data = {} 
    species = []
    coords = []
    energy_differences = []
    norms = []
    nacs = []

    log_filepaths = []
    
    def get_log_filenames(root_dir: str) -> List[str]:
        filenames = []
        for _, subdirs, _ in os.walk(root_dir):
            for subdir in subdirs:
                dir_ = os.fsencode(subdir)
                for file in os.listdir(dir_):
                    filename = os.fsdecode(file)
                    if filename.endswith('.log'):
                        filenames += [os.path.join(subdir, os.fsdecode(file))]
        return filenames

    log_filepaths += get_log_filenames(reactant_dir)
    log_filepaths += get_log_filenames(product_dir)

    for i, file in enumerate(log_filepaths):
        species_, coords_, energy_differences_, norms_, nacs_ = extract_data_from_file(file, natoms)        
        species += species_
        coords += coords_ 
        energy_differences += energy_differences_ 
        norms += norms_ 
        nacs += nacs_ 
        print(f'extracted data from file #{i + 1}: {file}')

    data['species'] = species
    data['coords'] = coords 
    data['e_diffs'] = energy_differences
    data['norms'] = norms
    data['nacs'] = nacs

    with open(write_file, 'w') as f:
        json.dump(data, f)
    print(f'successfully extracted data to `{write_file}`')
    print(f'size: {os.path.getsize(write_file)}')
        
def extract_data_from_file(data_file: str, natoms: int) -> NACData:
    with open(data_file) as f:
        raw_data = f.readlines()
        if is_happy_landing(raw_data):
            species, coords = extract_atom_types_and_coords(raw_data, natoms)
            energy_differences = extract_energy_differences(raw_data)
            norms = extract_norms(raw_data)
            nacs = extract_nacs(raw_data, natoms)
        else:
            warnings.warn(f'{data_file} is not valid')
            return [], [], [], [], [] # type: ignore

    return NACData(species, coords, energy_differences, norms, nacs)

def is_happy_landing(raw_data: List) -> bool:
    return 'Happy landing!' in raw_data[-4]

# TODO: better way to do this
def extract_atom_types_and_coords(data, natoms):
    atom_types = []
    coords = []
    for i, line in enumerate(data):
        if 'Cartesian coordinates in Angstrom:' in line:
            single_molecule_atom_types = []
            single_molecule_coords = []
            for j in range(4, natoms+4):
                split = data[i+j].split()
                atom_type = split[1][0]
                coord = [float(i) for i in split[2:]]
                single_molecule_atom_types.append(atom_type)
                single_molecule_coords.append(coord)
            atom_types.append(single_molecule_atom_types)
            coords.append(single_molecule_coords)

    return atom_types, coords
            
# TODO: better way to do this
def extract_energy_differences(data):
    energy_differences = []
    for _, line in enumerate(data):
        if 'Energy difference: ' in line:
            energy_difference = float(line.split('Energy difference: ')[1].replace('\n', ''))
            energy_differences.append(energy_difference)

    return energy_differences

# TODO: better way to do this
def extract_norms(data):
    norms = []
    for _, line in enumerate(data):
        if 'norm:' in line:
            match_number = re.compile(r'-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
            norm = [float(x) for x in re.findall(match_number, line)][0]
            norms.append(norm)

    return norms

# TODO: better way to do this
def extract_nacs(data, natoms):
    nacs = []
    for i, line in enumerate(data):
        if 'Total derivative coupling' in line:
            nac = []
            for j in range(8, natoms+8):
                match_number = re.compile(r'-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
                atom_nac = [float(x) for x in re.findall(match_number, data[i+j])][1:]
                nac.append(atom_nac)
            nacs.append(nac)

    return nacs


if __name__ == '__main__':
    extract_data_from_meci_logs(REACTANT_DIR, PRODUCT_DIR, WRITE_FILE, NATOMS)
