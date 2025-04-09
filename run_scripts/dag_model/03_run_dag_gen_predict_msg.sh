#python launcher_scripts/run_from_config.py configs/iceberg/dag_gen_predict_train_canopus.yaml
#python launcher_scripts/run_from_config.py configs/iceberg/dag_gen_predict_train_nist.yaml
# python launcher_scripts/run_from_config.py configs/iceberg/dag_gen_predict_train_msg_allev.yaml


# Assign intensities to prediction for next training run
python data_scripts/dag/add_dag_intens.py \
	--pred-dag-folder  results/dag_msg_allev/split_rnd1/preds_train_100/tree_preds.hdf5 \
	--true-dag-folder data/spec_datasets/msg/subformulae/no_subform.hdf5 \
	--out-dag-folder results/dag_msg_allev/split_rnd1/preds_train_100_inten.hdf5  \
	--num-workers 32 \
	--add-raw


python data_scripts/dag/add_dag_intens.py \
	--pred-dag-folder  results/dag_msg_allev/split_rnd2/preds_train_100/tree_preds.hdf5 \
	--true-dag-folder data/spec_datasets/msg/subformulae/no_subform.hdf5 \
	--out-dag-folder results/dag_msg_allev/split_rnd2/preds_train_100_inten.hdf5  \
	--num-workers 32 \
	--add-raw

python data_scripts/dag/add_dag_intens.py \
	--pred-dag-folder  results/dag_msg_allev/split_rnd3/preds_train_100/tree_preds.hdf5 \
	--true-dag-folder data/spec_datasets/msg/subformulae/no_subform.hdf5 \
	--out-dag-folder results/dag_msg_allev/split_rnd3/preds_train_100_inten.hdf5  \
	--num-workers 32 \
	--add-raw

