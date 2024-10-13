# maspr_server.py

import esm
import numpy as np
import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from metrics import fingerprint_projection
from optimal_fingerprint import label_to_featurization
from train_maspr import MorganPredictor
import subprocess

app = FastAPI()

class ESMFeaturizer:
    def __init__(self):
        self.model, self._alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self._alphabet.get_batch_converter()
        self.max_len = 1000

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def featurize(self, prot_seqs: List[str]) -> List[torch.Tensor]:
        """
        Featurize a list of protein sequences using the ESM model.

        Args:
            prot_seqs (List[str]): List of protein sequences.

        Returns:
            List[torch.Tensor]: List of embeddings for each sequence.
        """
        # Prepare sequences and labels for batch processing
        batch_data = []
        for i, seq in enumerate(prot_seqs):
            seq = seq.upper()[:self.max_len]
            batch_data.append((f'seq_{i}', seq))

        batch_labels, batch_strs, batch_tokens = self.batch_converter(batch_data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(tokens=batch_tokens, repr_layers=[33])
        token_representations = results["representations"][33]

        # Extract embeddings for each sequence
        embeddings = []
        for i, seq in enumerate(prot_seqs):
            seq_len = len(seq)
            tokens = token_representations[i, 1 : seq_len + 1]
            embeddings.append(tokens)

        return embeddings

class MASPR:
    def __init__(self, esm_featurizer=None, model_file='maspr_model.pt'):
        if esm_featurizer is None:
            self.esm_featurizer = ESMFeaturizer()
        else:
            self.esm_featurizer = esm_featurizer

        fpt_len = len(label_to_featurization['Val'])

        self.device = torch.device('cuda')
        self.model = MorganPredictor(10, 1280, fingerprint_len=fpt_len)

        state_dict = torch.load(model_file, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)

        self.model.eval()

        # Create index_to_label mapping
        self.labels = list(label_to_featurization.keys())
        self.index_to_label = {idx: label for idx, label in enumerate(self.labels)}

    def extract_stachelhaus(self, sequence):
        """
        Calls the external 'aden_predict' binary to extract stachelhaus residues.
        """
        try:
            result = subprocess.run(
                ['./aden_predict', 'stachelhaus-extract', '--query', sequence],
                capture_output=True,
                text=True,
                check=True
            )
            output_lines = result.stdout.strip().split('\n')
            aden_resi = [int(res) for res in output_lines[0].split(',') if res]
            # aden_resn = output_lines[1]  # If needed
            return aden_resi
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error in 'aden_predict': {e.stderr}")

    def extract_sequence(self, sequence):
        """
        Calls the external 'aden_predict' binary to extract A-domain sequences.
        """
        try:
            result = subprocess.run(
                ['./aden_predict', 'extract-sequence', '--query', sequence],
                capture_output=True,
                text=True,
                check=True
            )
            adom_seqs_and_pos = result.stdout.strip().split('\n')
            adom_seqs = [seq.split(',')[0] for seq in adom_seqs_and_pos if seq]
            return adom_seqs
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error in 'aden_predict': {e.stderr}")

    def extract_embedding(self, sequence):
        adom_seqs = self.extract_sequence(sequence)
        if not adom_seqs:
            raise ValueError("No A-domains found in the input sequence.")

        # Batch featurization of all A-domain sequences
        adom_embeddings = self.esm_featurizer.featurize(adom_seqs)

        embeds = []
        for adom_seq, adom_embedding in zip(adom_seqs, adom_embeddings):
            stach_residues = self.extract_stachelhaus(adom_seq)
            # Ensure indices are within bounds
            stach_residues = [idx for idx in stach_residues if idx < len(adom_embedding)]
            if not stach_residues:
                raise ValueError("No valid stachelhaus residues found.")
            embeds.append(adom_embedding[stach_residues])

        embeds = torch.stack(embeds)
        return embeds

    def predict(self, sequence, k):
        adom_embedding = self.extract_embedding(sequence).to(self.device)
        with torch.no_grad():
            pred_fpt, phi = self.model.forward(adom_embedding)
            sorted_outputs = fingerprint_projection(pred_fpt, self.model, device=self.device)
            predictions = sorted_outputs[:, :k]
            return predictions

# Instantiate the MASPR model once when the server starts
maspr_model = MASPR()

# Define request and response models
class PredictionRequest(BaseModel):
    sequence: str
    k: int

class PredictionResponse(BaseModel):
    predictions: List[List[str]]  # List of predictions for each A-domain

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        sequence, k = request.sequence, request.k
        predictions = maspr_model.predict(sequence, k)
        return PredictionResponse(predictions=predictions)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

