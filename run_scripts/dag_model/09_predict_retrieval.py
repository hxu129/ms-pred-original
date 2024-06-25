import yaml
from pathlib import Path
import subprocess
import json

pred_file = "src/ms_pred/dag_pred/predict_smis.py"
retrieve_file = "src/ms_pred/retrieval/rank_binned.py"
devices = ",".join(["1"])
max_nodes = 100
num_workers = 64
dist = "cos"

test_entries = [
    {"train_dataset": "nist20",
     "train_split": "split_1_rnd1",
     "test_dataset": "broad_distress",
     },
]


for test_entry in test_entries:
    train_dataset = test_entry['train_dataset']
    train_split =  test_entry['train_split']
    test_dataset = test_entry['test_dataset']
    inten_dir = Path(f"results/dag_inten_{train_dataset}")
    inten_model =  inten_dir / train_split  / "version_0/best.ckpt"
    if not inten_model.exists():
        print(f"Could not find model {inten_model}; skipping\n: {json.dumps(test_entry, indent=1)}")
        continue

    labels = f"data/elucidation/{test_dataset}/cands_df_{test_dataset}.tsv"
    spec_dir = Path(labels).parent / 'spec_files'

    save_dir = inten_model.parent.parent / f"elucidation_{test_dataset}"
    save_dir.mkdir(exist_ok=True)

    args = yaml.safe_load(open(inten_model.parent.parent / "args.yaml", "r"))
    form_folder = Path(args["magma_dag_folder"])
    gen_model = form_folder.parent / "version_0/best.ckpt"

    save_dir = save_dir
    save_dir.mkdir(exist_ok=True)
    cmd = f"""python {pred_file} \\
    --num-workers {num_workers}  \\
    --dataset-name {train_dataset} \\
    --sparse-out \\
    --sparse-k 100 \\
    --max-nodes {max_nodes} \\
    --gen-checkpoint {gen_model} \\
    --inten-checkpoint {inten_model} \\
    --save-dir {save_dir} \\
    --dataset-labels {labels} \\
    --binned-out \\
    """
    device_str = f"CUDA_VISIBLE_DEVICES={devices}"
    cmd = f"{device_str} {cmd}"
    print(cmd + "\n")
    subprocess.run(cmd, shell=True)

    # Run retrieval
    cmd = f"""python {retrieve_file} \\
    --spec-folder {spec_dir} \\
    --binned-pred-file {save_dir / 'binned_preds.p'} \\
    --dist-fn {dist} \\
    """
    print(cmd + "\n")
    subprocess.run(cmd, shell=True)
