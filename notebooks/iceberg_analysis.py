"""Script to submit ICEBERG calls, plot spectra and save fragmentation patterns."""

import datetime
import os
from typing import List, Dict

import pandas as pd
from rdkit import Chem

from ms_pred.dag_pred.iceberg_elucidation import (
    iceberg_prediction,
    explain_peaks_different_collision_energies,
)
from ms_pred.dag_pred.iceberg_extract_fragments import (
    analyze_fragmentation_patterns,
)

# Setup paths and environment
today = datetime.datetime.now().strftime('%Y%m%d')
SAVE_PATH = f'/home/magled/results/{today}'
os.makedirs(SAVE_PATH, exist_ok=True)
os.chdir('/home/magled/ms-pred-dev')

# Configuration
CONFIG = {
    'python_path': '/home/magled/miniconda3/envs/ms-gen-new/bin/python',
    'gen_ckpt': '/home/runzhong/ms-models/iceberg_results_20241111/dag_nist20/split_1_rnd1/version_0/best.ckpt',
    'inten_ckpt': '/home/runzhong/ms-models/iceberg_results_20241111/dag_inten_nist20/split_1_rnd1/version_1/best.ckpt',
    'cuda_devices': 1,
    'batch_size': 8,
    'num_workers': 32,
    'sparse_k': 100,
    'max_nodes': 100,
    'threshold': 0.0,
    'binned_out': False,
    'ppm': 20,
    'num_bins': 15000,
    'dist_func': 'entropy'
}

def count_atoms(smiles: str) -> int:
    """Count number of atoms in a molecule from SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return float('inf')  # Return infinity for invalid SMILES
        return mol.GetNumAtoms()
    except:
        return float('inf')

def get_dft_data(database_path: str, size: int = None) -> Dict[str, str]:
    """Get molecules with specified number of atoms from database."""
    dft_database = pd.read_csv(database_path)
    dft_database['atom_count'] = dft_database['molecule'].apply(count_atoms)
    if size is not None:
        molecules = dft_database[dft_database['atom_count'] == size]['molecule'].unique().tolist()
    else:
        molecules = dft_database['molecule'].unique().tolist()
    return {f'id_{i}': smiles for i, smiles in enumerate(molecules)}

def elucidate(
        smiles: str,
        collision_energies: List[int],
        file_name: str,
        save_path: str = SAVE_PATH,
        normalized_collision_energies: bool = True,
        num_peaks: int = 5,
        ) -> pd.DataFrame:
    """Run ICEBERG prediction and analysis for a single molecule."""
    print('=' * 50)
    print(f'Elucidating {file_name} with SMILES: {smiles}')
    print('=' * 50)

    # if SAVE_PATH/file_name.svg exists, skip
    if os.path.exists(f'{save_path}/{file_name}.svg'):
        print(f'{file_name}.svg already exists, skipping...')
        return

    # Run ICEBERG to predict spectra
    result_path, pmz = iceberg_prediction(
        candidate_smiles=[smiles],
        collision_energies=collision_energies,
        nce=normalized_collision_energies,
        save_path=save_path,
        **CONFIG
    )

    try:
        # Explain peaks at different collision energies
        explain_peaks_different_collision_energies(
            result_path=result_path,
            pmz=pmz,
            smiles=smiles,
            collision_energies=collision_energies,
            normalized_collision_energies=normalized_collision_energies,
            file_name=file_name,
            save_path=save_path,
            num_peaks=num_peaks
        )
    except ValueError as e:
        print(f'Error: {e}')
        # save SMILES and file_name to df which is called error.csv
        pd.DataFrame({'smiles': [smiles], 'file_name': [file_name]}).to_csv(f'{save_path}/error.csv', mode='a', header=False, index=False)
        return pd.DataFrame()

    # Analyze fragmentation patterns
    return analyze_fragmentation_patterns(
        result_path=result_path,
        smiles=smiles,
        collision_energies=collision_energies,
        normalized_collision_energies=normalized_collision_energies,
        file_name=file_name,
        save_path=save_path,
    )

def get_nist_test_data() -> Dict[str, str]:
    """
    Read NIST test data from TSV file and return a dictionary mapping NIST IDs to SMILES strings.
    
    Returns:
        Dict[str, str]: Dictionary with NIST IDs as keys and SMILES strings as values
    """
    nist20_dataset_path = '/home/runzhong/ms-pred/data/spec_datasets/nist20/labels.tsv'
    
    nist_dict = {}
    inchikeys = []
    with open(nist20_dataset_path, 'r') as f:
        # Skip header if present
        next(f)  # Skip the header line
        
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) >= 7:  # Ensure we have enough fields
                inchikey = fields[6]
                precursor_type = fields[4]
                if inchikey in inchikeys:
                    continue
                if precursor_type != '[M+H]+':
                    continue
                inchikeys.append(inchikey)
                spec_id = fields[1]    # spec ID is in second column
                smiles = fields[5]     # SMILES string is in sixth column
                nist_dict[spec_id] = smiles

    return nist_dict

def filter_for_test_data(molecules: Dict[str, str]) -> Dict[str, str]:
    """Filter molecules to only include those in the NIST test data."""

    splits_file = "/home/runzhong/ms-pred/data/spec_datasets/nist20/splits/split_1.tsv"

    with open(splits_file, 'r') as f:

        next(f)

        for line in f:
            fields = line.strip().split('\t')
            spec_id = fields[0]
            split_assignment = fields[1]

            if split_assignment != 'test':
                # filter this molecule out
                molecules.pop(spec_id, None)

    return molecules

def main():
    # Analysis parameters

    collision_energies = [5,10,20,40,80,100] # eV
    
    # If using NIST test data
    molecules = get_nist_test_data()
    molecules = filter_for_test_data(molecules)

    # If using DFT database
    # database_path = '/home/magled/external_files/rdf_data_190531.csv' # DFT database
    # molecules = get_dft_data(database_path)

    # If getting data from specific molecules instead, use this snippet:
    # collision_energies = [5, 9, 14, 20, 30, 40, 50, 65, 80, 95, 110, 130]  # eV, phenelzine
    # collision_energies = [30, 40, 50, 80]  # eV, pyrazole
    # collision_energies = [1, 9, 30] # eV, phosphonoacid
    # collision_energies = [10, 20, 30, 40, 50] # NCE, gabaarg
    # molecules = {
    #     # 'phenelzine': r'c1ccccc1CCNN'
    #     # 'pyrazole': r'CC1=C(C(=NN1)C)CCCl'
    #     # 'phosphonoacid': r'C(CC(=O)O)CP(=O)(O)O'
    #     # 'gabaarg': r'C(CC(C(=O)O)NC(=O)CCCN)CN=C(N)N'
    # }
    
    # Process all molecules
    for molecule_name, smiles in molecules.items():
        df = elucidate(
            smiles=smiles,
            collision_energies=collision_energies,
            file_name=molecule_name,
            save_path=SAVE_PATH,
            normalized_collision_energies=False,
            num_peaks=5
        )
    

if __name__ == '__main__':
    main()