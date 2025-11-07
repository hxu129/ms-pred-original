#!/usr/bin/env python
"""create_mixed_dataset.py

Script to merge Canopus and MassSpecGym datasets into a unified dataset.
"""
import pandas as pd
from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Source paths
CANOPUS_ROOT = Path("/local3/ericjiang/wgc/huaxu/ms/DiffMS/data/canopus")
MSG_ROOT = Path("/local3/ericjiang/wgc/huaxu/ms/DiffMS/data/msg")

# Output path
OUTPUT_ROOT = Path("data/spec_datasets/mixed_canopus_msg")


def create_mixed_labels():
    """Merge Canopus and MassSpecGym labels into a single file."""
    logging.info("Loading Canopus labels...")
    canopus_labels = pd.read_csv(CANOPUS_ROOT / "labels.tsv", sep="\t")
    logging.info(f"Loaded {len(canopus_labels)} Canopus entries")
    
    logging.info("Loading MassSpecGym labels...")
    msg_labels = pd.read_csv(MSG_ROOT / "labels.tsv", sep="\t")
    logging.info(f"Loaded {len(msg_labels)} MassSpecGym entries")
    
    # Ensure both have consistent columns
    # Canopus has: dataset, spec, name, ionization, formula, smiles, inchikey, instrument
    # MSG has: dataset, spec, ionization, formula, smiles, inchikey, instrument
    
    # Add 'name' column to MSG if missing
    if 'name' not in msg_labels.columns:
        msg_labels['name'] = ''
    
    # Ensure column order is consistent
    common_cols = ['dataset', 'spec', 'name', 'ionization', 'formula', 'smiles', 'inchikey']
    if 'instrument' in canopus_labels.columns and 'instrument' in msg_labels.columns:
        common_cols.append('instrument')
    
    canopus_subset = canopus_labels[common_cols]
    msg_subset = msg_labels[common_cols]
    
    # Merge the datasets
    logging.info("Merging datasets...")
    merged_labels = pd.concat([canopus_subset, msg_subset], ignore_index=True)
    logging.info(f"Merged dataset has {len(merged_labels)} total entries")
    
    # Save merged labels
    output_labels = OUTPUT_ROOT / "labels.tsv"
    output_labels.parent.mkdir(parents=True, exist_ok=True)
    merged_labels.to_csv(output_labels, sep="\t", index=False)
    logging.info(f"Saved merged labels to {output_labels}")
    
    return merged_labels


def create_mixed_split():
    """Merge Canopus and MassSpecGym splits into a unified split file."""
    logging.info("Loading Canopus split...")
    canopus_split = pd.read_csv(CANOPUS_ROOT / "splits/canopus_hplus_100_0.tsv", sep="\t")
    # Canopus split has columns: name, split
    # Rename 'name' to 'spec' if needed
    if 'name' in canopus_split.columns and 'spec' not in canopus_split.columns:
        canopus_split = canopus_split.rename(columns={'name': 'spec'})
    logging.info(f"Loaded {len(canopus_split)} Canopus split entries")
    
    logging.info("Loading MassSpecGym split...")
    msg_split = pd.read_csv(MSG_ROOT / "split.tsv", sep="\t")
    # MSG split has columns: name, split
    # Rename 'name' to 'spec' if needed
    if 'name' in msg_split.columns and 'spec' not in msg_split.columns:
        msg_split = msg_split.rename(columns={'name': 'spec'})
    logging.info(f"Loaded {len(msg_split)} MassSpecGym split entries")
    
    # Merge the splits
    logging.info("Merging splits...")
    merged_split = pd.concat([canopus_split, msg_split], ignore_index=True)
    logging.info(f"Merged split has {len(merged_split)} total entries")
    
    # Count train/test distribution
    train_count = (merged_split['split'] == 'train').sum()
    test_count = (merged_split['split'] == 'test').sum()
    val_count = (merged_split['split'] == 'val').sum()
    logging.info(f"Train: {train_count}, Test: {test_count}, Val: {val_count}")
    
    # Save merged split
    splits_dir = OUTPUT_ROOT / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    output_split = splits_dir / "split_1.tsv"
    merged_split.to_csv(output_split, sep="\t", index=False)
    logging.info(f"Saved merged split to {output_split}")
    
    return merged_split


def copy_subformulae_files():
    """Copy subformulae files directly instead of using symlinks."""
    import shutil
    
    logging.info("Copying subformulae files (this may take a few minutes)...")
    
    # Create subformulae directory
    subformulae_dir = OUTPUT_ROOT / "subformulae" / "mixed_subformulae"
    subformulae_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy Canopus subformulae
    canopus_subform_src = CANOPUS_ROOT / "subformulae/subformulae_default"
    logging.info(f"Copying Canopus subformulae from {canopus_subform_src}...")
    canopus_files = list(canopus_subform_src.glob("*.json"))
    for i, src_file in enumerate(canopus_files):
        if i % 1000 == 0:
            logging.info(f"  Copied {i}/{len(canopus_files)} Canopus files...")
        dst_file = subformulae_dir / src_file.name
        if not dst_file.exists():
            shutil.copy2(src_file, dst_file)
    logging.info(f"Copied {len(canopus_files)} Canopus subformulae files")
    
    # Copy MSG subformulae
    msg_subform_src = MSG_ROOT / "subformulae/default_subformulae"
    logging.info(f"Copying MSG subformulae from {msg_subform_src}...")
    msg_files = list(msg_subform_src.glob("*.json"))
    for i, src_file in enumerate(msg_files):
        if i % 10000 == 0:
            logging.info(f"  Copied {i}/{len(msg_files)} MSG files...")
        dst_file = subformulae_dir / src_file.name
        if not dst_file.exists():
            shutil.copy2(src_file, dst_file)
    logging.info(f"Copied {len(msg_files)} MSG subformulae files")
    
    total_files = len(list(subformulae_dir.glob("*.json")))
    logging.info(f"Total subformulae files: {total_files}")
    logging.info("Subformulae files copied successfully")


def main():
    """Main function to create the mixed dataset."""
    logging.info("Starting mixed dataset creation...")
    
    # Create merged labels
    merged_labels = create_mixed_labels()
    
    # Create merged split
    merged_split = create_mixed_split()
    
    # Copy subformulae files (instead of symlinks)
    copy_subformulae_files()
    
    logging.info("Mixed dataset creation completed successfully!")
    logging.info(f"Output directory: {OUTPUT_ROOT}")
    logging.info("Note: Spec files are still in original locations - access them via labels")
    

if __name__ == "__main__":
    main()

