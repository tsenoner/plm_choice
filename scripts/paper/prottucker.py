#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python script_name.py -i input.fasta -m model_dir --save_cath --save_embeddings

import argparse
import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import h5py
import numpy as np
import torch
from torch import nn
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORMER_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"

class NonStandardAminoAcidError(Exception):
    """Exception raised for non-standard amino acids in the sequence."""
    pass

class TuckerFNN(nn.Module):
    def __init__(self):
        super(TuckerFNN, self).__init__()
        self.tucker = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
        )

    def single_pass(self, x: torch.Tensor) -> torch.Tensor:
        return self.tucker(x)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ancor = self.single_pass(X[:, 0, :])
        pos = self.single_pass(X[:, 1, :])
        neg = self.single_pass(X[:, 2, :])
        return ancor, pos, neg

class ProtTucker:
    ROSTLAB_BASE_URL = "https://rostlab.org/~deepppi"
    CATH_BASE_FILENAME = "cath_v430_dom_seqs_S100_161121"
    PROTTUCKER_WEIGHTS_URL = f"{ROSTLAB_BASE_URL}/embedding_repo/embedding_models/ProtTucker/ProtTucker_ProtT5.pt"
    EAT_DBS_BASE_URL = f"{ROSTLAB_BASE_URL}/eat_dbs"
    CATH_LABELS_URL = f"{EAT_DBS_BASE_URL}/{CATH_BASE_FILENAME}_labels.txt"
    CATH_EMBEDDINGS_URL = f"{EAT_DBS_BASE_URL}/{CATH_BASE_FILENAME}.npy"
    CATH_IDS_URL = f"{EAT_DBS_BASE_URL}/{CATH_BASE_FILENAME}.txt"

    def __init__(self, model_dir: Path):
        self.prottucker_dir = model_dir / "prottucker"
        self.prottucker_dir.mkdir(exist_ok=True)
        self.model = self._load_model()
        self.lookup_labels = self._read_annotation()
        self.lookup_ids, self.lookup_embs = self._read_prottucker_embs()

    def _load_model(self) -> TuckerFNN:
        checkpoint_p = self.prottucker_dir / "tucker_weights.pt"
        model = TuckerFNN()
        return load_model(model, self.PROTTUCKER_WEIGHTS_URL, checkpoint_p)

    def _read_annotation(self) -> Dict[str, str]:
        label_p = self.prottucker_dir / f"{self.CATH_BASE_FILENAME}_labels.txt"
        if not label_p.is_file():
            download_file(self.CATH_LABELS_URL, label_p)
        with open(label_p, 'r') as in_f:
            return {line.strip().split(',')[0]: line.strip().split(',')[1] for line in in_f}

    def _read_prottucker_embs(self) -> Tuple[List[str], torch.Tensor]:
        emb_local_p = self.prottucker_dir / f"{self.CATH_BASE_FILENAME}.npy"
        if not emb_local_p.is_file():
            download_file(self.CATH_EMBEDDINGS_URL, emb_local_p)
        embeddings = torch.from_numpy(np.load(emb_local_p)).to(DEVICE)
        embeddings = self.model.single_pass(embeddings)

        id_local_p = self.prottucker_dir / f"{self.CATH_BASE_FILENAME}.txt"
        if not id_local_p.is_file():
            download_file(self.CATH_IDS_URL, id_local_p)
        with open(id_local_p, 'r') as in_f:
            ids = [line.strip().split("|")[-1].split("_")[0] for line in in_f]
        return ids, embeddings.unsqueeze(dim=0)

    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply ProtTucker model to the input embeddings."""
        with torch.no_grad():
            return self.model.single_pass(embeddings)

    def write_embeddings(self, embeddings: torch.Tensor, ids: List[str], out_path: Path):
        """Write ProtTucker embeddings to an HDF5 file."""
        with h5py.File(out_path, 'w') as hf:
            for idx, protein_id in enumerate(ids):
                hf.create_dataset(protein_id, data=embeddings[idx].cpu().numpy())
        print(f"ProtTucker embeddings saved to {out_path}")

    def write_predictions(self, predictions: List[Tuple[str, str, float]], out_path: Path):
        with open(out_path, "w") as out_f:
            out_f.write("\n".join([f"{lookup_id}\t{cath_anno}\t{nn_dist:.3f}"
                                   for lookup_id, cath_anno, nn_dist in predictions]))

class ProtT5Embedder:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.prott5 = None
        self.tokenizer = None

    def _load_model(self):
        if self.prott5 is None or self.tokenizer is None:
            print("Loading ProtT5...")
            start = time.time()
            prott5_dir = self.model_dir / "prott5"
            self.prott5 = T5EncoderModel.from_pretrained(TRANSFORMER_NAME, torch_dtype=torch.float16,
                                                         cache_dir=prott5_dir)
            self.prott5 = self.prott5.to(DEVICE).eval()
            self.tokenizer = T5Tokenizer.from_pretrained(TRANSFORMER_NAME, do_lower_case=False, cache_dir=prott5_dir, legacy=False)
            print(f"Finished loading {TRANSFORMER_NAME} in {time.time()-start:.1f}[s]")

    def _process_batch(self, batch: List[Tuple[str, str, int]]) -> List[Tuple[str, torch.Tensor]]:
        pdb_ids, seqs, seq_lens = zip(*batch)

        token_encoding = self.tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding='longest')
        input_ids = torch.tensor(token_encoding['input_ids']).to(DEVICE)
        attention_mask = torch.tensor(token_encoding['attention_mask']).to(DEVICE)

        try:
            with torch.no_grad():
                prott5_output = self.prott5(input_ids, attention_mask=attention_mask)
        except RuntimeError:
            print("RuntimeError for {} (L={})".format(pdb_ids, seq_lens))
            return []

        residue_embedding = prott5_output.last_hidden_state.detach()
        residue_embedding = residue_embedding * attention_mask.unsqueeze(dim=-1)

        return [(pdb_id, residue_embedding[idx, :seq_len].mean(dim=0))
                for idx, (pdb_id, seq_len) in enumerate(zip(pdb_ids, seq_lens))]

    def embed_sequences(self, seq_dict: Dict[str, str], max_batch_size: int = 100, max_residues: int = 4000) -> List[Tuple[str, torch.Tensor]]:
        self._load_model()
        sorted_seqs = sorted(seq_dict.items(), key=lambda kv: len(kv[1]), reverse=True)
        batch = []
        embeddings = []
        print("Generating protein embeddings...")
        start = time.time()

        for seq_idx, (seq_id, seq) in tqdm(enumerate(sorted_seqs, 1)):
            seq_len = len(seq)
            seq = ' '.join(list(seq))
            batch.append((seq_id, seq, seq_len))
            n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len

            if (len(batch) >= max_batch_size or n_res_batch >= max_residues or seq_idx == len(sorted_seqs)):
                embeddings.extend(self._process_batch(batch))
                batch = []

        exe_time = time.time() - start
        print(f'Total time for generating embeddings: {exe_time:.2f} [s] ### Avg. time per protein: {exe_time/len(embeddings):.3f} [s]')
        return embeddings

    def read_embeddings_from_h5(self, h5_path: Path) -> List[Tuple[str, torch.Tensor]]:
        """Read pre-computed ProtT5 embeddings from an HDF5 file."""
        embeddings = []
        with h5py.File(h5_path, 'r') as hf:
            for protein_id in hf.keys():
                embedding = torch.tensor(hf[protein_id][:], dtype=torch.float16).to(DEVICE)
                embeddings.append((protein_id, embedding))
        print(f"Read {len(embeddings)} pre-computed embeddings from {h5_path}")
        return embeddings

    def write_embeddings_to_h5(self, embeddings: List[Tuple[str, torch.Tensor]], out_path: Path):
        """Write ProtT5 embeddings to an HDF5 file."""
        with h5py.File(out_path, 'w') as hf:
            for protein_id, embedding in embeddings:
                hf.create_dataset(protein_id, data=embedding.cpu().numpy())
        print(f"ProtT5 embeddings saved to {out_path}")

def eat(lookup_embs: torch.Tensor, lookup_ids: List[str], lookup_labels: Dict[str, str],
        queries: torch.Tensor, threshold: float = None, norm: int = 2) -> List[Tuple[str, str, float]]:
    pdist = torch.cdist(lookup_embs, queries.unsqueeze(dim=0), p=norm).squeeze(dim=0)
    nn_dists, nn_idxs = torch.topk(pdist, 1, largest=False, dim=0)
    predictions = []
    for test_idx in range(queries.shape[0]):
        nn_idx = int(nn_idxs[:, test_idx])
        nn_dist = float(nn_dists[:, test_idx])
        lookup_id = lookup_ids[nn_idx]
        if threshold is not None and nn_dist > threshold:
            lookup_id = "N/A"
            lookup_label = "N/A"
        else:
            lookup_label = lookup_labels[lookup_id]
        predictions.append((lookup_id, lookup_label, nn_dist))
    return predictions

def download_file(url: str, local_path: Path):
    from urllib import request
    import shutil

    if not local_path.parent.is_dir():
        local_path.parent.mkdir(parents=True)
    print(f"Downloading: {url}")
    req = request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with request.urlopen(req) as response, open(local_path, 'wb') as outfile:
        shutil.copyfileobj(response, outfile)

def load_model(model: nn.Module, weights_link: str, checkpoint_p: Path, state_dict: str = "state_dict") -> nn.Module:
    if not checkpoint_p.exists():
        download_file(weights_link, checkpoint_p)
    state = torch.load(checkpoint_p, map_location=DEVICE)
    model.load_state_dict(state[state_dict])
    return model.eval().half().to(DEVICE)

def read_fasta(fasta_path: Path) -> Dict[str, str]:
    """Read a FASTA file and return a dictionary of sequences."""
    valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")
    translation_table = str.maketrans("BZJ", "XXX")

    def parse_fasta(fasta_file: Iterator[str]) -> Iterator[tuple[str, str]]:
        """Parse FASTA file and yield (id, sequence) tuples."""
        current_id, current_seq = None, []
        for line in map(str.strip, fasta_file):
            if line.startswith('>'):
                if current_id:
                    yield current_id, ''.join(current_seq)
                current_id, current_seq = line[1:], []
            elif current_id:
                current_seq.append(line)
        if current_id:
            yield current_id, ''.join(current_seq)

    def validate_seq(seq: str, seq_id: str) -> str:
        """Validate and clean the sequence."""
        seq = seq.translate(translation_table).upper()
        if invalid_aa := set(seq) - valid_aa:
            raise NonStandardAminoAcidError(f"Non-standard amino acid(s) {', '.join(invalid_aa)} found in sequence {seq_id}")
        return seq

    with open(fasta_path, 'r') as fasta_file:
        sequences = {seq_id: validate_seq(seq, seq_id) for seq_id, seq in parse_fasta(fasta_file)}

    print(f"Read {len(sequences)} sequences from {fasta_path}")
    return sequences

def main():
    parser = argparse.ArgumentParser(description='ProtTucker predictions for protein sequences.')
    parser.add_argument('-i', '--input', required=True, type=str, help='Path to a fasta-formatted file containing protein sequences.')
    parser.add_argument('-m', '--model_dir', required=True, type=str, help='Directory for storing model checkpoints.')
    parser.add_argument('--prott5_embeddings', type=str, help='Path to pre-computed ProtT5 embeddings in HDF5 format.')

    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument('--save_cath', action='store_true', help='Save CATH predictions')
    output_group.add_argument('--save_embeddings', action='store_true', help='Save ProtTucker embeddings')
    output_group.add_argument('--save_both', action='store_true', help='Save both CATH predictions and ProtTucker embeddings')
    args = parser.parse_args()

    input_path = Path(args.input)
    model_dir = Path(args.model_dir)
    output_dir = input_path.parent
    base_name = input_path.stem

    prottucker = ProtTucker(model_dir)
    embedder = ProtT5Embedder(model_dir)

    if args.prott5_embeddings:
        # Use pre-computed ProtT5 embeddings
        embeddings = embedder.read_embeddings_from_h5(Path(args.prott5_embeddings))
    else:
        # Compute ProtT5 embeddings
        seq_dict = read_fasta(input_path)
        embeddings = embedder.embed_sequences(seq_dict)

        # Save computed ProtT5 embeddings
        prott5_embeddings_path = output_dir / f"{base_name}_prott5.h5"
        embedder.write_embeddings_to_h5(embeddings, prott5_embeddings_path)
        print(f"Computed ProtT5 embeddings saved to {prott5_embeddings_path}")

    print("Applying ProtTucker model...")
    protein_ids, protein_embeddings = zip(*embeddings)
    protein_embeddings = torch.stack(protein_embeddings)
    prottucker_embeddings = prottucker.predict(protein_embeddings)

    if args.save_embeddings or args.save_both:
        embeddings_path = output_dir / f"{base_name}_prottucker_embeddings.h5"
        prottucker.write_embeddings(prottucker_embeddings, protein_ids, embeddings_path)

    if args.save_cath or args.save_both:
        print("Making ProtTucker CATH predictions...")
        predictions = eat(
            prottucker.lookup_embs,
            prottucker.lookup_ids,
            prottucker.lookup_labels,
            prottucker_embeddings
        )

        cath_predictions_path = output_dir / f"{base_name}_prottucker_cath_predictions.txt"
        prottucker.write_predictions(predictions, cath_predictions_path)
        print(f"CATH predictions written to {cath_predictions_path}")


if __name__ == '__main__':
    main()
