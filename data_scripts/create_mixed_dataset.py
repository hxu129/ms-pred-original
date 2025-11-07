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


def create_symlinks():
    """Create symlinks for spec_files and subformulae directories."""
    logging.info("Creating directory structure with symlinks...")
    
    # Create spec_files directory
    spec_files_dir = OUTPUT_ROOT / "spec_files"
    spec_files_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subformulae directory
    subformulae_dir = OUTPUT_ROOT / "subformulae"
    subformulae_dir.mkdir(parents=True, exist_ok=True)
    
    # Create symlink for Canopus spec files
    canopus_spec_link = spec_files_dir / "canopus_specs"
    if not canopus_spec_link.exists():
        os.symlink(CANOPUS_ROOT / "spec_files", canopus_spec_link)
        logging.info(f"Created symlink: {canopus_spec_link} -> {CANOPUS_ROOT / 'spec_files'}")
    
    # Create symlink for MSG spec files
    msg_spec_link = spec_files_dir / "msg_specs"
    if not msg_spec_link.exists():
        os.symlink(MSG_ROOT / "spec_files", msg_spec_link)
        logging.info(f"Created symlink: {msg_spec_link} -> {MSG_ROOT / 'spec_files'}")
    
    # Create symlink for Canopus subformulae
    canopus_subform_link = subformulae_dir / "canopus_subformulae"
    if not canopus_subform_link.exists():
        os.symlink(CANOPUS_ROOT / "subformulae/subformulae_default", canopus_subform_link)
        logging.info(f"Created symlink: {canopus_subform_link} -> {CANOPUS_ROOT / 'subformulae/subformulae_default'}")
    
    # Create symlink for MSG subformulae
    msg_subform_link = subformulae_dir / "msg_subformulae"
    if not msg_subform_link.exists():
        os.symlink(MSG_ROOT / "subformulae/default_subformulae", msg_subform_link)
        logging.info(f"Created symlink: {msg_subform_link} -> {MSG_ROOT / 'subformulae/default_subformulae'}")
    
    logging.info("Symlinks created successfully")


def main():
    """Main function to create the mixed dataset."""
    logging.info("Starting mixed dataset creation...")
    
    # Create merged labels
    merged_labels = create_mixed_labels()
    
    # Create merged split
    merged_split = create_mixed_split()
    
    # Create symlinks for data files
    create_symlinks()
    
    logging.info("Mixed dataset creation completed successfully!")
    logging.info(f"Output directory: {OUTPUT_ROOT}")
    

if __name__ == "__main__":
    main()

