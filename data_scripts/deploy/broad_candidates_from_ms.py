"""
Build retrieval list
"""
import functools

import pandas as pd
import subprocess
import argparse
import pickle
from datetime import datetime
from pathlib import Path
from platformdirs import user_cache_dir
from tqdm import tqdm
from ms_pred import common
import re
import numpy as np
from rdkit import Chem


def get_args():
    """get_args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pickle-file", default='data/retrieval/pubchem/pubchem_formula_map.p'
    )
    parser.add_argument(
        "--spec-dir", default='data/elucidation/broad_distress/spec_files'
    )
    parser.add_argument(
        '--max-formula', default=5, type=int,
    )
    return parser.parse_args()


def sirius_get_formula_candidate(spec_dir, max_formula=5, ppm_max=10, profile='orbitrap'):
    """
    Call sirius to get the formula candidates
    Args:
        spec_dir:
        max_formula: max number of formula candidates
        ppm_max: max ppm for ms1 and ms2
        profile: possible options: `default`, 'qtof', 'orbitrap', 'fticr'

    Returns:
        {spec_name: formula_candidates}
    """
    ms_files = spec_dir.glob('*.ms')
    name_to_spec = {_.stem: _ for _ in ms_files}

    # sirius writes output to the filesystem
    out_dir = Path(user_cache_dir(f"ms-pred/sirius-out-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    sirius_command = (f'sirius -o {out_dir} '
                      f'-i {spec_dir} '
                      f'--ignore-formula --noCite formula '
                      f'-p {profile} --ppm-max={ppm_max} write-summaries')
    print("Running SIRIUS, command:\n" + sirius_command + "\n")
    subprocess.run(sirius_command, shell=True)

    spec_to_info = {}
    for pred_dir in tqdm(out_dir.glob('*/'), total=len(name_to_spec)):
        if not pred_dir.is_dir():
            continue
        spec_name = pred_dir.stem.split('_')[-1]
        assert spec_name in name_to_spec

        # read from SIRIUS output
        formula_pred_path = pred_dir / 'formula_candidates.tsv'
        if formula_pred_path.exists():
            df = pd.read_csv(formula_pred_path, sep='\t')
            candidates = df.loc[df['formulaRank'] <= max_formula, ['molecularFormula', 'adduct']].values.tolist()

            # remove spaces
            for cand in candidates:
                cand[1] = cand[1].replace(' ', '')

            # remove unsupported adduct
            new_candidates = []
            for cand in candidates:
                if cand[1] in common.ion2mass:
                    new_candidates.append(cand)
                else:
                    print(f'In {spec_name}, skipping {cand} because adduct type is not supported')
            candidates = new_candidates

            # get collision energies
            spec_file = name_to_spec[spec_name]
            meta, spec = common.parse_spectra(spec_file)
            precursor = meta['parentmass']
            colli_engs = []
            for (k, v) in spec:
                colli_engs.append(common.get_collision_energy(k))

            spec_to_info[spec_name] = {
                'spec_name': spec_name, 'formula_adduct': candidates, 'collision_energies': colli_engs, 'precursor': precursor
            }
        else:
            print(f'Formula prediction failed for {spec_name}, no formula found in ppm_max={ppm_max}')

    print(f'{len(spec_to_info)} out of {len(name_to_spec)} specs has predicted formula')
    return spec_to_info


def main():
    args = get_args()
    spec_dir = args.spec_dir
    pickle_file = args.pickle_file
    max_formula = args.max_formula

    spec_dir = Path(spec_dir)
    dataset_name = spec_dir.parent.stem
    output_df = spec_dir / f'../cands_df_{dataset_name}.tsv'

    # Generate the candidates
    spec_to_info = sirius_get_formula_candidate(spec_dir, max_formula=max_formula)

    # OR: define your own candidates
    # spec_to_info = {
    #     'GABA-Arg': {
    #         'spec_name': 'GABA-Arg',
    #         'formula_adduct': [('C10H21N5O3', '[M+H]+')],
    #         'collision_energies': ['10', '20', '30', '40', '50'],
    #         'precursor': 260.171715984,
    #     },
    # }

    print('Loading pubchem map')
    full_map = pickle.load(open(pickle_file, "rb"))
    uniq_forms = set()
    for info in spec_to_info.values():
        for form_adduct in info['formula_adduct']:
            uniq_forms.add(form_adduct[0]) # get form
    sub_map = {uniq_form: full_map.get(uniq_form, []) for uniq_form in uniq_forms}

    for spec, info in spec_to_info.items():
        info['formula'] = []
        info['adduct'] = []
        info['isomers'] = []
        for form, adduct in info['formula_adduct']:
            all_isomers = sub_map.get(form, [])
            if len(all_isomers) == 0:
                continue

            # Filter down to ensure only unique isomers
            all_isomers_ars = np.array([(i, j) for i, j in all_isomers])
            _, uniq_inds = np.unique(all_isomers_ars[:, 1], return_index=True)

            # All isomers (smi, inchikey)
            all_isomers = {
                (i, j) for i, j in all_isomers_ars[uniq_inds]
                if '.' not in i # remove multiple molecules
            }
            info['formula'].append(form)
            info['adduct'].append(adduct)
            info['isomers'].append(all_isomers)
        del info['formula_adduct'] # delete old key


    print("Dumping to df")
    # Build retrieval list
    def build_list(info):
        entries = []
        spec_name = info['spec_name']
        for adduct, all_isomers in zip(info['adduct'], info['isomers']):
            for cand_smi, cand_inchikey in all_isomers:
                # Validating SMILES by roundtripping it
                try:
                    mol = Chem.MolFromSmiles(cand_smi)
                    if mol is None:
                        continue
                    inchi = Chem.MolToInchi(mol)
                    mol = common.canonical_mol_from_inchi(inchi)
                    if mol is None:
                        continue
                except RuntimeError:
                    continue

                new_entry = {
                    "spec": spec_name,
                    "smiles": cand_smi,
                    "ionization": adduct,
                    "inchikey": cand_inchikey,
                    "precursor": info['precursor'],
                    "collision_energies": info['collision_energies'],
                }
                entries.append(new_entry)
        return entries

    entries = common.chunked_parallel(list(spec_to_info.values()), build_list, chunks=1000, max_cpu=32)
    # entries = [build_list(v) for v in tqdm(spec_to_info.values())]
    merged_entries = []
    for e in entries:
        merged_entries += e

    df_out = pd.DataFrame(merged_entries)
    df_out.to_csv(
        output_df,
        sep="\t",
        index=None,
    )


if __name__ == '__main__':
    main()
