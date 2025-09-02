python launcher_scripts/run_from_config.py configs/iceberg/dag_inten_train_nist20.yaml

# contrastive finetune
# you have to download https://zenodo.org/records/15529765/files/pubchem_formulae_inchikey.hdf5
# and place it at data/pubchem/pubchem_formulae_inchikey.hdf5
python launcher_scripts/run_from_config.py configs/iceberg/dag_inten_contr_finetune_nist20.yaml

# MassSpecGym:
# To train with entropy loss:
# python launcher_scripts/run_from_config.py configs/iceberg/dag_inten_train_msg_allev_entropy.yaml
# To train with cosine loss:
# python launcher_scripts/run_from_config.py configs/iceberg/dag_inten_train_msg_allev_cosine.yaml
