"""
Analyze formulae vocabulary coverage for GraffMS models.

This script helps determine the optimal vocabulary size by analyzing
how many formulas are needed to explain what percentage of signal intensity.

Usage:
    python analysis/analyze_vocab_coverage.py \
        --dataset canopus_train_public \
        --split-name split_1.tsv \
        --form-dir-name magma_subform_50_with_raw \
        --vocab-sizes 1000,2500,5000,7500,10000,15000
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import logging

import sys
sys.path.append('src')

from ms_pred.graff_ms import graff_ms_data
import ms_pred.nn_utils as nn_utils
import ms_pred.common as common


def get_args():
    parser = argparse.ArgumentParser(description='Analyze vocabulary coverage')
    parser.add_argument('--dataset', default='canopus_train_public', 
                       help='Dataset name')
    parser.add_argument('--split-name', default='split_1.tsv',
                       help='Split file name')
    parser.add_argument('--form-dir-name', default='magma_subform_50_with_raw',
                       help='Subformula directory name')
    parser.add_argument('--vocab-sizes', default='1000,2500,5000,7500,10000,15000',
                       help='Comma-separated list of vocabulary sizes to test')
    parser.add_argument('--output-dir', default='analysis/vocab_analysis',
                       help='Output directory for plots and results')
    return parser.parse_args()


def analyze_formula_distribution(dataset):
    """Analyze the distribution of formulas in the dataset."""
    top_forms = dataset.get_top_forms()
    forms = top_forms['forms']
    counts = top_forms['cts']
    
    total_count = counts.sum()
    
    print("\n" + "="*60)
    print("Formula Distribution Analysis")
    print("="*60)
    print(f"Total unique formulas: {len(forms):,}")
    print(f"Total formula occurrences: {total_count:,}")
    print(f"\nTop 10 most frequent formulas:")
    for i in range(min(10, len(forms))):
        freq = counts[i] / total_count * 100
        print(f"  {i+1}. Count: {counts[i]:>6} ({freq:>5.2f}%)")
    
    # Cumulative coverage
    cumsum = np.cumsum(counts) / total_count * 100
    
    milestones = [50, 80, 90, 95, 98, 99, 99.5]
    print(f"\nCoverage milestones:")
    for milestone in milestones:
        idx = np.searchsorted(cumsum, milestone)
        if idx < len(cumsum):
            print(f"  {milestone:>5.1f}% coverage: {idx+1:>6,} formulas " + 
                  f"({(idx+1)/len(forms)*100:.2f}% of unique formulas)")
    
    return forms, counts, total_count


def analyze_intensity_coverage(dataset, vocab_sizes):
    """
    Analyze what percentage of intensity is covered by top-K formulas.
    This is more meaningful than just counting formula occurrences.
    """
    print("\n" + "="*60)
    print("Intensity Coverage Analysis")
    print("="*60)
    
    # Collect all formulas with their intensities
    formula_intensities = defaultdict(float)
    total_intensity = 0.0
    
    for spec_name in dataset.spec_names:
        spec_data = dataset.name_to_forms[spec_name]
        
        # Extract formulas and their intensities
        if 'formulae' in spec_data and 'binned' in spec_data:
            formulas = spec_data['formulae']
            intensities = spec_data['binned']
            
            for form_arr, inten in zip(formulas, intensities):
                # Convert array to tuple for hashing
                form_key = tuple(form_arr)
                formula_intensities[form_key] += inten
                total_intensity += inten
    
    print(f"Total intensity across all spectra: {total_intensity:.2f}")
    print(f"Unique formulas observed: {len(formula_intensities):,}")
    
    # Sort by intensity
    sorted_formulas = sorted(formula_intensities.items(), 
                            key=lambda x: x[1], reverse=True)
    
    # Compute coverage for different vocab sizes
    results = []
    for vocab_size in vocab_sizes:
        covered_intensity = sum(inten for _, inten in sorted_formulas[:vocab_size])
        coverage_pct = covered_intensity / total_intensity * 100
        
        results.append({
            'vocab_size': vocab_size,
            'coverage_pct': coverage_pct,
            'num_unique_formulas': len(formula_intensities),
            'vocab_pct_of_unique': vocab_size / len(formula_intensities) * 100
        })
        
        print(f"\nVocab size: {vocab_size:>6,}")
        print(f"  Intensity coverage: {coverage_pct:>6.2f}%")
        print(f"  % of unique formulas: {vocab_size/len(formula_intensities)*100:>6.2f}%")
    
    return results, sorted_formulas


def plot_coverage_curves(results, output_dir):
    """Plot coverage curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    vocab_sizes = [r['vocab_size'] for r in results]
    coverages = [r['coverage_pct'] for r in results]
    
    # Plot 1: Coverage vs Vocab Size
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale
    axes[0].plot(vocab_sizes, coverages, 'o-', linewidth=2, markersize=8)
    axes[0].axhline(y=95, color='r', linestyle='--', label='95% coverage')
    axes[0].axhline(y=98, color='g', linestyle='--', label='98% coverage (GraffMS)')
    axes[0].set_xlabel('Vocabulary Size', fontsize=12)
    axes[0].set_ylabel('Intensity Coverage (%)', fontsize=12)
    axes[0].set_title('Coverage vs Vocabulary Size', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Log scale
    axes[1].semilogx(vocab_sizes, coverages, 'o-', linewidth=2, markersize=8)
    axes[1].axhline(y=95, color='r', linestyle='--', label='95% coverage')
    axes[1].axhline(y=98, color='g', linestyle='--', label='98% coverage (GraffMS)')
    axes[1].set_xlabel('Vocabulary Size (log scale)', fontsize=12)
    axes[1].set_ylabel('Intensity Coverage (%)', fontsize=12)
    axes[1].set_title('Coverage vs Vocabulary Size (Log Scale)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'coverage_curves.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot: {output_dir / 'coverage_curves.png'}")
    
    # Plot 2: Marginal gains
    fig, ax = plt.subplots(figsize=(10, 6))
    marginal_gains = [coverages[0]] + [coverages[i] - coverages[i-1] 
                                       for i in range(1, len(coverages))]
    
    ax.bar(range(len(vocab_sizes)), marginal_gains, 
           tick_label=[f'{v//1000}K' for v in vocab_sizes],
           color='steelblue', alpha=0.7)
    ax.set_xlabel('Vocabulary Size', fontsize=12)
    ax.set_ylabel('Marginal Coverage Gain (%)', fontsize=12)
    ax.set_title('Diminishing Returns of Larger Vocabularies', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'marginal_gains.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {output_dir / 'marginal_gains.png'}")


def save_results(results, output_dir):
    """Save results to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    df = pd.DataFrame(results)
    output_file = output_dir / 'vocab_coverage_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved results: {output_file}")
    
    return df


def main():
    args = get_args()
    
    # Parse vocab sizes
    vocab_sizes = [int(x.strip()) for x in args.vocab_sizes.split(',')]
    
    print("\n" + "="*60)
    print("GraffMS Vocabulary Coverage Analysis")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split_name}")
    print(f"Formula dir: {args.form_dir_name}")
    print(f"Vocabulary sizes to test: {vocab_sizes}")
    
    # Load dataset
    dataset_name = args.dataset
    data_dir = Path("data/spec_datasets") / dataset_name
    labels = data_dir / "labels.tsv"
    split_file = data_dir / "splits" / args.split_name
    
    df = pd.read_csv(labels, sep="\t")
    spec_names = df["spec"].values
    train_inds, _, _ = common.get_splits(spec_names, split_file)
    train_df = df.iloc[train_inds]
    
    print(f"\nLoading training data ({len(train_df)} spectra)...")
    
    # Load subformulae
    subform_stem = args.form_dir_name
    subformula_folder = data_dir / "subformulae" / subform_stem
    form_map = {i.stem: Path(i) for i in subformula_folder.glob("*.json")}
    
    # Create graph featurizer
    graph_featurizer = nn_utils.MolDGLGraph(pe_embed_k=0)
    
    # Create dataset
    train_dataset = graff_ms_data.BinnedDataset(
        train_df,
        form_map=form_map,
        data_dir=data_dir,
        num_bins=15000,
        upper_limit=1500,
        graph_featurizer=graph_featurizer,
    )
    
    print(f"✓ Loaded {len(train_dataset)} training spectra")
    
    # Analyze formula distribution
    forms, counts, total_count = analyze_formula_distribution(train_dataset)
    
    # Analyze intensity coverage
    results, sorted_formulas = analyze_intensity_coverage(train_dataset, vocab_sizes)
    
    # Save results
    df_results = save_results(results, args.output_dir)
    
    # Plot curves
    plot_coverage_curves(results, args.output_dir)
    
    # Print recommendations
    print("\n" + "="*60)
    print("Recommendations")
    print("="*60)
    
    # Find vocab size for 95% and 98% coverage
    for target in [95, 98]:
        for r in results:
            if r['coverage_pct'] >= target:
                print(f"For {target}% coverage: {r['vocab_size']:,} formulas needed")
                break
    
    print("\n✓ Analysis complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

