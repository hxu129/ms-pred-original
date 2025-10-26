""" joint_model. """
from collections import defaultdict
import numpy as np
import torch
import pytorch_lightning as pl
from rdkit import Chem

import ms_pred.common as common
import ms_pred.magma.fragmentation as fragmentation
import ms_pred.dag_pred.gen_model as gen_model
import ms_pred.dag_pred.inten_model as inten_model
import ms_pred.dag_pred.dag_data as dag_data


class JointModel(pl.LightningModule):
    def __init__(
        self,
        gen_model_obj: gen_model.FragGNN,
        inten_model_obj: inten_model.IntenGNN,
    ):
        """__init__.

        Args:
            gen_model_obj (gen_model.FragGNN): gen_model_obj
            inten_model_obj (inten_model.IntenGNN): inten_model_obj
        """

        super().__init__()
        self.gen_model_obj = gen_model_obj
        self.inten_model_obj = inten_model_obj
        self.inten_collate_fn = dag_data.IntenPredDataset.get_collate_fn()

        root_enc_gen = self.gen_model_obj.root_encode
        pe_embed_gen = self.gen_model_obj.pe_embed_k
        add_hs_gen = self.gen_model_obj.add_hs

        root_enc_inten = self.inten_model_obj.root_encode
        pe_embed_inten = self.inten_model_obj.pe_embed_k
        add_hs_inten = self.inten_model_obj.add_hs

        self.gen_tp = dag_data.TreeProcessor(
            root_encode=root_enc_gen, pe_embed_k=pe_embed_gen, add_hs=add_hs_gen
        )

        self.inten_tp = dag_data.TreeProcessor(
            root_encode=root_enc_inten, pe_embed_k=pe_embed_inten, add_hs=add_hs_inten
        )

    @classmethod
    def from_checkpoints(cls, gen_checkpoint, inten_checkpoint, map_location="cpu"):
        """from_checkpoints.

        Args:
            gen_checkpoint: Path to generation model checkpoint
            inten_checkpoint: Path to intensity model checkpoint
            map_location: Device to load models to (default: "cpu")
        """

        gen_model_obj = gen_model.FragGNN.load_from_checkpoint(gen_checkpoint, map_location=map_location)
        inten_model_obj = inten_model.IntenGNN.load_from_checkpoint(inten_checkpoint, map_location=map_location)
        return cls(gen_model_obj, inten_model_obj)

    def predict_mol(
        self,
        mol: Chem.Mol,
        smi: str,
        adduct: str,
        threshold: float,
        device: str,
        max_nodes: int,
        binned_out: bool = False,
    ):
        """predict_mol.

        Args:
            smi (str): smi
            adduct
            threshold (float): threshold
            device (str): device
            max_nodes (int): max_nodes
            binned_out
        """

        self.eval()
        self.freeze()

        # Run tree gen model
        # Defines exact tree
        root_smi = smi
        root_mol = mol
        root_inchi = common.inchi_from_smiles(root_smi)

        frag_tree = self.gen_model_obj.predict_mol(
            root_mol=root_mol,
            root_smi=root_smi,
            adduct=adduct,
            threshold=threshold,
            device=device,
            max_nodes=max_nodes,
        )
        frag_tree = {"root_inchi": root_inchi, "root_smi": root_smi, "root_mol": root_mol, "name": "", "frags": frag_tree}

        # Get engine from fragmentation for this inchi
        engine = fragmentation.FragmentEngine(mol_str=smi, root_mol=root_mol, mol_str_type="smiles")

        processed_tree = self.inten_tp.process_tree_inten_pred(frag_tree)

        # Save for output wrangle
        out_tree = processed_tree["tree"]
        processed_tree = processed_tree["dgl_tree"]

        processed_tree["adduct"] = common.ion2onehot_pos[adduct]
        processed_tree["name"] = ""
        batch = self.inten_collate_fn([processed_tree])
        inten_frag_ids = batch["inten_frag_ids"]

        safe_device = lambda x: x.to(device) if x is not None else x

        frag_graphs = safe_device(batch["frag_graphs"])
        root_reprs = safe_device(batch["root_reprs"])
        ind_maps = safe_device(batch["inds"])
        num_frags = safe_device(batch["num_frags"])
        broken_bonds = safe_device(batch["broken_bonds"])
        max_remove_hs = safe_device(batch["max_remove_hs"])
        max_add_hs = safe_device(batch["max_add_hs"])
        masses = safe_device(batch["masses"])

        adducts = safe_device(batch["adducts"]).to(device)
        root_forms = safe_device(batch["root_form_vecs"])
        frag_forms = safe_device(batch["frag_form_vecs"])

        # IDs to use to recapitulate
        inten_preds = self.inten_model_obj.predict(
            graphs=frag_graphs,
            root_reprs=root_reprs,
            ind_maps=ind_maps,
            num_frags=num_frags,
            max_breaks=broken_bonds,
            max_add_hs=max_add_hs,
            max_remove_hs=max_remove_hs,
            masses=masses,
            root_forms=root_forms,
            frag_forms=frag_forms,
            binned_out=binned_out,
            adducts=adducts,
        )

        if binned_out:
            out = inten_preds
        else:
            inten_preds = inten_preds["spec"][0]
            inten_frag_ids = inten_frag_ids[0]
            out_frags = out_tree["frags"]

            # Get masses too
            for inten_pred, inten_frag_id in zip(inten_preds, inten_frag_ids):
                out_frags[inten_frag_id]["intens"] = inten_pred.tolist()

                new_masses = (
                    out_frags[inten_frag_id]["base_mass"] + engine.shift_bucket_masses
                )
                mz_with_charge = new_masses + common.ion2mass[adduct]
                out_frags[inten_frag_id]["mz_no_charge"] = new_masses.tolist()
                out_frags[inten_frag_id]["mz_charge"] = mz_with_charge.tolist()

            out_tree["frags"] = out_frags
            out = out_tree
        return out

    def predict_mol_batch(
        self,
        mol_list: list,
        smi_list: list,
        adduct_list: list,
        threshold: float,
        device: str,
        max_nodes: int,
        binned_out: bool = False,
    ):
        """predict_mol_batch.
        
        Batch prediction for multiple molecules. Generates fragmentation trees
        sequentially (unavoidable due to autoregressive generation), then batches
        the intensity prediction for GPU efficiency.

        Args:
            mol_list: List of RDKit molecules
            smi_list: List of SMILES strings
            adduct_list: List of adduct strings
            threshold: Threshold for fragment generation
            device: Device to run on
            max_nodes: Max nodes per fragmentation tree
            binned_out: Whether to return binned output

        Returns:
            List of prediction dictionaries (one per molecule)
        """
        self.eval()
        self.freeze()
        
        # Ensure deterministic behavior
        torch.set_grad_enabled(False)

        if len(mol_list) == 0:
            return []

        # Step 1: Generate fragmentation trees sequentially (can't be batched)
        frag_trees = []
        engines = []
        
        # Step 2: Process all trees and prepare for batched intensity prediction
        processed_trees = []
        out_trees = []
        
        for mol, smi, adduct in zip(mol_list, smi_list, adduct_list):
            root_mol = mol
            root_smi = smi
            root_inchi = common.inchi_from_smiles(root_smi)

            # Generate fragmentation tree (autoregressive, must be sequential)
            frag_tree_dict = self.gen_model_obj.predict_mol(
                root_mol=root_mol,
                root_smi=root_smi,
                adduct=adduct,
                threshold=threshold,
                device=device,
                max_nodes=max_nodes,
            )
            
            frag_tree = {
                "root_inchi": root_inchi,
                "root_smi": root_smi,
                "root_mol": root_mol,
                "name": "",
                "frags": frag_tree_dict
            }
            
            # Create engine for this molecule
            engine = fragmentation.FragmentEngine(
                mol_str=smi, root_mol=root_mol, mol_str_type="smiles"
            )
            engines.append(engine)

            processed = self.inten_tp.process_tree_inten_pred(frag_tree)
            out_trees.append(processed["tree"])
            
            processed_dgl = processed["dgl_tree"]
            processed_dgl["adduct"] = common.ion2onehot_pos[adduct]
            processed_dgl["name"] = ""
            processed_trees.append(processed_dgl)

        # Step 3: Collate all processed trees into a single batch
        batch = self.inten_collate_fn(processed_trees)
        inten_frag_ids_batch = batch["inten_frag_ids"]

        safe_device = lambda x: x.to(device) if x is not None else x

        frag_graphs = safe_device(batch["frag_graphs"])
        root_reprs = safe_device(batch["root_reprs"])
        ind_maps = safe_device(batch["inds"])
        num_frags = safe_device(batch["num_frags"])
        broken_bonds = safe_device(batch["broken_bonds"])
        max_remove_hs = safe_device(batch["max_remove_hs"])
        max_add_hs = safe_device(batch["max_add_hs"])
        masses = safe_device(batch["masses"])
        adducts_tensor = safe_device(batch["adducts"]).to(device)
        root_forms = safe_device(batch["root_form_vecs"])
        frag_forms = safe_device(batch["frag_form_vecs"])

        # Step 4: Run batched intensity prediction (KEY OPTIMIZATION!)
        inten_preds = self.inten_model_obj.predict(
            graphs=frag_graphs,
            root_reprs=root_reprs,
            ind_maps=ind_maps,
            num_frags=num_frags,
            max_breaks=broken_bonds,
            max_add_hs=max_add_hs,
            max_remove_hs=max_remove_hs,
            masses=masses,
            root_forms=root_forms,
            frag_forms=frag_forms,
            binned_out=binned_out,
            adducts=adducts_tensor,
        )

        # Step 5: Unpack batch results back to individual predictions
        outputs = []
        
        if binned_out:
            # For binned output, split the batch
            for i in range(len(mol_list)):
                outputs.append(inten_preds[i])
        else:
            # Unpack fragment-level predictions
            inten_preds_spec = inten_preds["spec"]
            
            for i, (out_tree, engine, adduct) in enumerate(zip(out_trees, engines, adduct_list)):
                inten_pred = inten_preds_spec[i]
                inten_frag_ids = inten_frag_ids_batch[i]
                out_frags = out_tree["frags"]

                # Assign intensities and masses to each fragment
                for pred, frag_id in zip(inten_pred, inten_frag_ids):
                    out_frags[frag_id]["intens"] = pred.tolist()

                    new_masses = (
                        out_frags[frag_id]["base_mass"] + engine.shift_bucket_masses
                    )
                    mz_with_charge = new_masses + common.ion2mass[adduct]
                    out_frags[frag_id]["mz_no_charge"] = new_masses.tolist()
                    out_frags[frag_id]["mz_charge"] = mz_with_charge.tolist()

                out_tree["frags"] = out_frags
                outputs.append(out_tree)

        return outputs
