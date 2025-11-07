""" Spectrum prediction evaluation with configurable binning

Use to compare binned predictions to ground truth spec values
Supports different binning strategies for evaluation

"""
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
import yaml
import pickle
from collections import defaultdict
from functools import partial
from numpy.linalg import norm
from scipy.stats import sem

import ms_pred.common as common


def cos_sim_fn(pred_ar, true_spec):
    """cos_sim_fn with normalization."""
    norm_pred = max(norm(pred_ar), 1e-6)
    norm_true = max(norm(true_spec), 1e-6)
    cos_sim = np.dot(pred_ar, true_spec) / (norm_pred * norm_true)
    return cos_sim


def rebin_spectrum(spectrum, original_bins, target_bins, upper_limit):
    """
    Rebin a spectrum from original_bins to target_bins.
    
    Args:
        spectrum: 1D array of shape [original_bins]
        original_bins: Number of bins in the input spectrum
        target_bins: Desired number of bins for output
        upper_limit: Upper m/z limit (e.g., 1500)
    
    Returns:
        Rebinned spectrum of shape [target_bins]
    """
    if original_bins == target_bins:
        return spectrum
    
    # Convert binned spectrum to peak list
    original_bin_width = upper_limit / original_bins
    peaks = []
    for idx, inten in enumerate(spectrum):
        if inten > 0:
            mz = idx * original_bin_width
            peaks.append([mz, inten])
    
    if len(peaks) == 0:
        return np.zeros(target_bins)
    
    peaks = np.array(peaks)
    
    # Rebin to target_bins
    rebinned = common.bin_spectra([peaks], num_bins=target_bins, upper_limit=upper_limit, pool_fn="add")
    return rebinned[0]


def get_args():
    """get_args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="canopus_train_public")
    parser.add_argument("--formula-dir-name", default="subform_20")
    parser.add_argument("--binned-pred-file")
    parser.add_argument("--outfile", default=None)
    parser.add_argument(
        "--min-inten",
        type=float,
        default=1e-5,
        help="Minimum intensity to call a peak in prediction",
    )
    parser.add_argument(
        "--max-peaks", type=int, default=20, help="Max num peaks to call"
    )
    parser.add_argument(
        "--eval-bins",
        type=int,
        default=None,
        help="Number of bins for evaluation (default: use model's bins). "
             "Examples: 150 for 10 Da bins, 1500 for 1 Da bins, 15000 for 0.1 Da bins"
    )
    parser.add_argument(
        "--eval-upper-limit",
        type=float,
        default=None,
        help="Upper m/z limit for evaluation (default: use model's upper_limit)"
    )
    return parser.parse_args()


def process_spec_file(spec_name, num_bins: int, upper_limit: int, spec_dir: Path):
    """process_spec_file."""
    spec_file = spec_dir / f"{spec_name}.json"
    loaded_json = json.load(open(spec_file, "r"))

    if loaded_json.get("output_tbl") is None:
        return None

    # Load without adduct involved
    mz = loaded_json["output_tbl"]["formula_mass_no_adduct"]
    inten = loaded_json["output_tbl"]["ms2_inten"]
    spec_ar = np.vstack([mz, inten]).transpose(1, 0)
    binned = common.bin_spectra([spec_ar], num_bins, upper_limit, pool_fn="add")
    avged = binned[0]
    return avged


def main(args):
    """main."""
    dataset = args.dataset
    formula_dir_name = args.formula_dir_name
    data_folder = Path(f"data/spec_datasets/{dataset}/subformulae/{formula_dir_name}/")
    min_inten = args.min_inten
    max_peaks = args.max_peaks

    binned_pred_file = Path(args.binned_pred_file)
    outfile = args.outfile
    if outfile is None:
        outfile = binned_pred_file.parent / "pred_eval.yaml"
        outfile_grouped = binned_pred_file.parent / "pred_eval_grouped.tsv"

    # Load predictions
    pred_specs = pickle.load(open(binned_pred_file, "rb"))
    pred_spec_ars = pred_specs["preds"]
    pred_smiles = pred_specs["smiles"]
    pred_spec_names = pred_specs["spec_names"]
    upper_limit = pred_specs["upper_limit"]
    num_bins = pred_specs["num_bins"]
    
    print(f"Model predictions: {num_bins} bins, upper_limit={upper_limit}")

    # Determine evaluation bins
    eval_bins = args.eval_bins if args.eval_bins is not None else num_bins
    eval_upper_limit = args.eval_upper_limit if args.eval_upper_limit is not None else upper_limit
    
    print(f"Evaluation binning: {eval_bins} bins, upper_limit={eval_upper_limit}")
    if eval_bins != num_bins:
        bin_resolution = eval_upper_limit / eval_bins
        print(f"  -> Bin resolution: {bin_resolution:.2f} Da/bin")
        print(f"  -> Rebinning predictions from {num_bins} to {eval_bins} bins")

    # Load ground truth spectra
    read_spec = partial(
        process_spec_file,
        num_bins=eval_bins,  # Use evaluation bins
        upper_limit=eval_upper_limit,
        spec_dir=data_folder,
    )
    true_specs = common.chunked_parallel(
        pred_spec_names, read_spec, max_cpu=16, chunks=100, timeout=60
    )

    # Rebin predictions if necessary
    if eval_bins != num_bins:
        print("Rebinning predictions...")
        rebinned_preds = []
        for pred_spec in pred_spec_ars:
            rebinned = rebin_spectrum(pred_spec, num_bins, eval_bins, eval_upper_limit)
            rebinned_preds.append(rebinned)
        pred_spec_ars = np.array(rebinned_preds)
        print(f"Rebinned predictions shape: {pred_spec_ars.shape}")

    # Compute similarities
    all_cos_sims = []
    matched_inds = []

    for ind, (pred_spec, spec_name, true_spec) in enumerate(
        zip(pred_spec_ars, pred_spec_names, true_specs)
    ):
        if true_spec is None:
            continue

        cos_sim = cos_sim_fn(pred_spec, true_spec)
        all_cos_sims.append(cos_sim)
        matched_inds.append(ind)

    all_cos_sims = np.array(all_cos_sims)
    
    # Compute statistics
    output = {
        "mean_cos_sim": float(all_cos_sims.mean()),
        "median_cos_sim": float(np.median(all_cos_sims)),
        "std_cos_sim": float(all_cos_sims.std()),
        "num_examples": len(all_cos_sims),
        "eval_bins": int(eval_bins),
        "eval_upper_limit": float(eval_upper_limit),
        "eval_bin_resolution_da": float(eval_upper_limit / eval_bins),
        "model_bins": int(num_bins),
        "model_upper_limit": float(upper_limit),
    }

    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(f"Mean Cosine Similarity: {output['mean_cos_sim']:.4f}")
    print(f"Median Cosine Similarity: {output['median_cos_sim']:.4f}")
    print(f"Std Cosine Similarity: {output['std_cos_sim']:.4f}")
    print(f"Number of Examples: {output['num_examples']}")
    print(f"Evaluation Bin Resolution: {output['eval_bin_resolution_da']:.2f} Da")
    print("="*50)

    # Save results
    with open(outfile, "w") as fp:
        yaml.dump(output, fp)
    print(f"\nResults saved to: {outfile}")

    return output


if __name__ == "__main__":
    args = get_args()
    main(args)

