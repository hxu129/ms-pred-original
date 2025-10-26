""" joint_model. """
from collections import defaultdict
import numpy as np
import torch
import time
import pytorch_lightning as pl
from rdkit import Chem
import logging
import multiprocessing as mp
from functools import partial
import ms_pred.common as common
import ms_pred.magma.fragmentation as fragmentation
import ms_pred.dag_pred.gen_model as gen_model
import ms_pred.dag_pred.inten_model as inten_model
import ms_pred.dag_pred.dag_data as dag_data

# In joint_model.py, _process_single_molecule_worker function
import warnings
warnings.filterwarnings('ignore', message='.*non-writable.*')


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

    @staticmethod
    def _process_single_molecule_worker(args):
        """
        Worker function for multiprocessing fragmentation tree generation.
        Must be static/top-level to be picklable.
        
        Args:
            args: Tuple of (mol, smi, adduct, gen_model_cpu, inten_tp, threshold, max_nodes)
            
        Returns:
            Tuple of (engine, processed_tree, out_tree)
        """
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
        mol, smi, adduct, gen_model_cpu, inten_tp, threshold, max_nodes = args
        
        try:
            root_mol = mol
            root_smi = smi
            root_inchi = common.inchi_from_smiles(root_smi)
            
            # Generate fragmentation tree on CPU
            frag_tree_dict = gen_model_cpu.predict_mol(
                root_mol=root_mol,
                root_smi=root_smi,
                adduct=adduct,
                threshold=threshold,
                device='cpu',
                max_nodes=max_nodes,
            )
            
            frag_tree = {
                "root_inchi": root_inchi,
                "root_smi": root_smi,
                "root_mol": root_mol,
                "name": "",
                "frags": frag_tree_dict
            }
            
            # Create engine
            engine = fragmentation.FragmentEngine(
                mol_str=smi, root_mol=root_mol, mol_str_type="smiles"
            )
            
            # Process tree
            processed = inten_tp.process_tree_inten_pred(frag_tree)
            out_tree = processed["tree"]
            
            processed_dgl = processed["dgl_tree"]
            processed_dgl["adduct"] = common.ion2onehot_pos[adduct]
            processed_dgl["name"] = ""
            
            return (engine, processed_dgl, out_tree)
        except Exception as e:
            logging.error(f"Error processing molecule {smi}: {e}")
            return None

    def predict_mol_batch_parallel(
        self,
        mol_list: list,
        smi_list: list,
        adduct_list: list,
        threshold: float,
        device: str,
        max_nodes: int,
        binned_out: bool = False,
        num_workers: int = None,
    ):
        """predict_mol_batch with multiprocessing for fragmentation trees.
        
        Parallelizes the fragmentation tree generation across CPU cores,
        then batches the intensity prediction on GPU for efficiency.
        
        Key optimization: Fragmentation (slow, autoregressive) runs in parallel on CPU,
        while intensity prediction (fast, batched) runs on GPU.
        
        Args:
            mol_list: List of RDKit molecules
            smi_list: List of SMILES strings
            adduct_list: List of adduct strings
            threshold: Threshold for fragment generation
            device: Device for intensity prediction (GPU)
            max_nodes: Max nodes per fragmentation tree
            binned_out: Whether to return binned output
            num_workers: Number of CPU workers (default: len(mol_list) or CPU count)
            
        Returns:
            List of prediction dictionaries (one per molecule)
        """
        self.eval()
        self.freeze()
        
        # Ensure deterministic behavior
        torch.set_grad_enabled(False)
        
        if len(mol_list) == 0:
            return []
        
        # Determine number of workers
        if num_workers is None:
            num_workers = min(len(mol_list), mp.cpu_count())
        
        logging.info(f"Using {num_workers} CPU workers for parallel fragmentation tree generation")
        
        # Step 1: Generate fragmentation trees in parallel on CPU
        curr_time = time.time()
        
        # Create CPU copy of gen_model for workers
        # NOTE: Each worker will get a copy via serialization
        gen_model_cpu = self.gen_model_obj.to('cpu')
        
        # Prepare arguments for workers
        worker_args = [
            (mol, smi, adduct, gen_model_cpu, self.inten_tp, threshold, max_nodes)
            for mol, smi, adduct in zip(mol_list, smi_list, adduct_list)
        ]
        
        # Process in parallel using multiprocessing
        engines = []
        processed_trees = []
        out_trees = []
        
        # Use spawn method to avoid CUDA fork issues
        curr_time = time.time()
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=num_workers) as pool:
            results = pool.map(JointModel._process_single_molecule_worker, worker_args)

        logging.info(f"Fragmentation tree processing time (parallel): {time.time() - curr_time:.2f} seconds")
        # Move gen_model back to target device
        # self.gen_model_obj.to(device)

        # Collect results
        for result in results:
            if result is not None:
                engine, processed_dgl, out_tree = result
                engines.append(engine)
                processed_trees.append(processed_dgl)
                out_trees.append(out_tree)
        
        logging.info(f"Fragmentation tree processing time (parallel): {time.time() - curr_time:.2f} seconds")
        
        if len(processed_trees) == 0:
            return []
        
        # Step 2: Collate all processed trees into a single batch
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
        
        # Step 3: Run batched intensity prediction on GPU (KEY OPTIMIZATION!)
        curr_time = time.time()
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
        logging.info(f"Intensity prediction time: {time.time() - curr_time:.2f} seconds")
        
        # Step 4: Unpack batch results back to per-molecule predictions
        curr_time = time.time()
        outputs = []
        
        if binned_out:
            # For binned output, split the batch
            for i in range(len(mol_list)):
                outputs.append(inten_preds[i])
        else:
            # Unpack fragment-level predictions (matching sequential version exactly)
            inten_preds_spec = inten_preds["spec"]
            
            for mol_idx, (out_tree, engine, adduct) in enumerate(zip(out_trees, engines, adduct_list)):
                # Get fragment predictions for this molecule (matching sequential version)
                inten_pred = inten_preds_spec[mol_idx]
                inten_frag_ids = inten_frag_ids_batch[mol_idx]
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
        
        logging.info(f"Unpack batch results time: {time.time() - curr_time:.2f} seconds")
        
        return outputs

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
        curr_time = time.time()
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

        logging.info(f"Fragmentation tree processing time: {time.time() - curr_time:.2f} seconds")

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
        curr_time = time.time()
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
        logging.info(f"Intensity prediction time: {time.time() - curr_time:.2f} seconds")

        # Step 5: Unpack batch results back to individual predictions
        outputs = []
        curr_time = time.time()
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

        logging.info(f"Unpack batch results time: {time.time() - curr_time:.2f} seconds")

        return outputs
