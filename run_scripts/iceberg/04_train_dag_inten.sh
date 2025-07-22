python launcher_scripts/run_from_config.py configs/iceberg/dag_inten_train_nist20.yaml

# contrastive finetune
python launcher_scripts/run_from_config.py configs/iceberg/dag_inten_contr_finetune_nist20.yaml

# MassSpecGym:
# To train with entropy loss:
# python launcher_scripts/run_from_config.py configs/iceberg/dag_inten_train_msg_allev_entropy.yaml
# To train with cosine loss:
# python launcher_scripts/run_from_config.py configs/iceberg/dag_inten_train_msg_allev_cosine.yaml
