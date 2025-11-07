"""
Example script showing how to evaluate GraffMS predictions with different binning strategies.

This demonstrates that you can:
1. Train with 15000 bins (0.1 Da resolution) - fixed in model
2. Evaluate with different bin sizes (e.g., 150, 1500, 15000) to test robustness
"""
from pathlib import Path
import subprocess

# Configuration
num_workers = 32
python_file = "src/ms_pred/graff_ms/predict.py"
eval_file = "analysis/spec_pred_eval_with_binning.py"  # New evaluation script
devices = "0"

# Test configuration
dataset_name = "canopus_train_public"
split = "split_4"
folder = "split_4_rnd1"

# Model path
res_folder = Path(f"results/graff_ms_baseline_{dataset_name}")
model = res_folder / f"{folder}/version_1/best.ckpt"
save_dir = model.parent.parent / "preds"
save_dir.mkdir(exist_ok=True, parents=True)

# Step 1: Make predictions (always uses model's 15000 bins)
print("="*80)
print("Step 1: Making predictions with model (15000 bins)")
print("="*80)

pred_cmd = f"""CUDA_VISIBLE_DEVICES={devices} python {python_file} \\
    --batch-size 32 \\
    --dataset-name {dataset_name} \\
    --split-name {split}.tsv \\
    --num-workers {num_workers} \\
    --subset-datasets test_only \\
    --checkpoint {model} \\
    --save-dir {save_dir} \\
    --gpu"""

print(pred_cmd)
subprocess.run(pred_cmd, shell=True)

out_binned = save_dir / "binned_preds.p"

# Step 2: Evaluate with different binning strategies
print("\n" + "="*80)
print("Step 2: Evaluating with different binning strategies")
print("="*80)

binning_strategies = [
    {
        "name": "Fine (0.1 Da bins)",
        "eval_bins": 15000,  # Same as model
        "comment": "Standard evaluation - same as training"
    },
    {
        "name": "Medium (1 Da bins)", 
        "eval_bins": 1500,
        "comment": "More realistic for typical MS instruments"
    },
    {
        "name": "Coarse (10 Da bins)",
        "eval_bins": 150,
        "comment": "Very coarse binning for robustness testing"
    },
    {
        "name": "Coarse (15 Da bins)",
        "eval_bins": 100,
        "comment": "Very coarse binning for robustness testing"
    },
    {
        "name": "Coarse (20 Da bins)",
        "eval_bins": 75,
        "comment": "Very coarse binning for robustness testing"
    },
    {
        "name": "Coarse (30 Da bins)",
        "eval_bins": 50,
        "comment": "Very coarse binning for robustness testing"
    },
]

results = {}

for strategy in binning_strategies:
    name = strategy["name"]
    eval_bins = strategy["eval_bins"]
    comment = strategy["comment"]
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {name}")
    print(f"Comment: {comment}")
    print(f"{'='*80}")
    
    eval_cmd = f"""python {eval_file} \\
        --binned-pred-file {out_binned} \\
        --max-peaks 100 \\
        --min-inten 0 \\
        --formula-dir-name no_subform \\
        --dataset {dataset_name} \\
        --eval-bins {eval_bins} \\
        --eval-upper-limit 1500.0"""
    
    print(eval_cmd)
    result = subprocess.run(eval_cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    
    # Store results
    results[name] = {
        "bins": eval_bins,
        "resolution_da": 1500.0 / eval_bins,
    }

# Summary
print("\n" + "="*80)
print("SUMMARY: Comparison of Different Binning Strategies")
print("="*80)
print(f"{'Strategy':<30} {'Bins':<10} {'Resolution (Da)':<20}")
print("-"*80)
for name, info in results.items():
    print(f"{name:<30} {info['bins']:<10} {info['resolution_da']:<20.2f}")
print("="*80)

print("\n📝 Note: All strategies evaluate the SAME model predictions (15000 bins)")
print("   The binning is only applied during evaluation metric computation.")
print("\n💡 Key Insight: Using coarser bins (e.g., 1-10 Da) during evaluation")
print("   can better reflect real-world MS instrument precision and show")
print("   if your model is robust to small m/z shifts.")

