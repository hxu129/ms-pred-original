"""predict.py

Make predictions with trained model

"""

import logging
from datetime import datetime
import yaml
import argparse
import json
import ast
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

import torch
import pytorch_lightning as pl

import ms_pred.common as common
import ms_pred.dag_pred.gen_model as gen_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--batch-size", default=64, action="store", type=int)
    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_tree_pred/")

    parser.add_argument(
        "--checkpoint-pth",
        help="name of checkpoint file",
        default="results/debug_dag_canopus_train_public/split_1/version_0/best.ckpt",
    )
    parser.add_argument("--dataset-name", default="gnps2015_debug")
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument("--split-name", default="split_22.tsv")
    parser.add_argument(
        "--subset-datasets",
        default="none",
        action="store",
        choices=["none", "train_only", "test_only", "debug_special"],
    )
    parser.add_argument("--threshold", default=0.5, action="store", type=float)
    parser.add_argument("--max-nodes", default=100, action="store", type=int)
    return parser.parse_args()


def predict():
    args = get_args()
    kwargs = args.__dict__

    save_dir = kwargs["save_dir"]
    common.setup_logger(save_dir, log_name="dag_gen_pred.log", debug=kwargs["debug"])
    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # Dump args
    yaml_args = yaml.dump(kwargs)
    logging.info(f"\n{yaml_args}")
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    # Get dataset
    # Load smiles dataset and split into 3 subsets
    dataset_name = kwargs["dataset_name"]
    data_dir = common.get_data_dir(dataset_name)
    labels = data_dir / kwargs["dataset_labels"]

    # Get train, val, test inds
    df = pd.read_csv(labels, sep="\t")

    if kwargs["subset_datasets"] != "none":
        splits = pd.read_csv(data_dir / "splits" / kwargs["split_name"], sep="\t")
        folds = set(splits.keys())
        folds.remove("spec")
        fold_name = list(folds)[0]
        if kwargs["subset_datasets"] == "train_only":
            names = splits[splits[fold_name] == "train"]["spec"].tolist()
        elif kwargs["subset_datasets"] == "test_only":
            names = splits[splits[fold_name] == "test"]["spec"].tolist()
        elif kwargs["subset_datasets"] == "debug_special":
            names = ["CCMSLIB00000577858"]
        else:
            raise NotImplementedError()

        df = df[df["spec"].isin(names)]

    # Create model and load
    best_checkpoint = kwargs["checkpoint_pth"]

    # Load from checkpoint
    model = gen_model.FragGNN.load_from_checkpoint(best_checkpoint, map_location="cpu")

    logging.info(f"Loaded model with from {best_checkpoint}")
    save_path = Path(kwargs["save_dir"]) / "tree_preds.hdf5"
    save_path.parent.mkdir(exist_ok=True)
    with torch.no_grad():
        model.eval()
        model.freeze()
        gpu = kwargs["gpu"]
        device = "cuda" if gpu else "cpu"
        model.to(device)

        def single_predict_mol(entry):
            torch.set_num_threads(1)

            smi = entry["smiles"]
            name = entry["spec"]
            adduct = entry["ionization"]
            precursor_mz = entry["precursor"]
            collision_energies = [i for i in ast.literal_eval(entry["collision_energies"])]
            inchi = Chem.MolToInchi(Chem.MolFromSmiles(smi))
            return_dict = {}
            for colli_eng in collision_energies:
                colli_eng_val = float(colli_eng.split()[0])
                pred = model.predict_mol(
                    smi,
                    precursor_mz=precursor_mz,
                    collision_eng=colli_eng_val,
                    adduct=adduct,
                    threshold=kwargs["threshold"],
                    device=device,
                    max_nodes=kwargs["max_nodes"],
                )
                output = {
                    "root_inchi": inchi,
                    "name": name,
                    "frags": pred,
                    "collision_energy": colli_eng_val,
                }
                out_name = f"pred_{name}_collision {colli_eng}.json"
                return_dict[out_name] = json.dumps(output, indent=2)
            return return_dict

        entries = [j for _, j in df.iterrows()]
        if kwargs["debug"]:
            entries = entries[:10]
            kwargs["batch_size"] = 0

        if kwargs["batch_size"] == 0:
            output_dicts = [single_predict_mol(i) for i in tqdm(entries)]
        else:
            output_dicts = common.chunked_parallel(
                entries,
                single_predict_mol,
                chunks=1000,
                max_cpu=kwargs["batch_size"],
            )

        # Write all output jsons to a hdf5 file
        h5 = common.HDF5Dataset(save_path, mode='w')
        for out_dic in tqdm(output_dicts):
            h5.write_dict(out_dic)
        h5.close()


if __name__ == "__main__":
    import time

    start_time = time.time()
    predict()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
