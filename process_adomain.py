# Given an input A-domain sequence, this script should extract the adomain sequence,

import argparse
import esm
import numpy as np
import subprocess
import torch

from metrics import fingerprint_projection
from optimal_fingerprint import label_to_featurization
from train_maspr import MorganPredictor

class ESMFeaturizer:
    def __init__(self):
        self.model, self._alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self._alphabet.get_batch_converter() # Converts amino acids to alphabet indices
        self.max_len = 1000

        # generate sequences on cpu for now cause this is too big for my gpu
        self.device = torch.device('cuda')
        self.model = self.model.to(self.device)

    def featurize(self, prot_seq: str):
        """
        prot_seq: the protein sequence to featurize

        returns: the ESM embedding of the protein sequence with shape (n, 1280)
        """
        prot_seq = prot_seq.upper()[:self.max_len]
        batch_labels, batch_strs, batch_tokens = self.batch_converter( [('sequence', prot_seq)] )
        batch_tokens = batch_tokens.to(self.device)
        results = self.model(tokens=batch_tokens, repr_layers=[33])
        token_representations = results["representations"][33].detach()
        tokens = token_representations[0, 1:len(prot_seq) + 1]
        return tokens

class MASPR:
    def __init__(self, esm_featurizer=None, model_file=None):
        if esm_featurizer is None:
            self.esm_featurizer = ESMFeaturizer()
        else:
            self.esm_featurizer = esm_featurizer

        fpt_len = len(label_to_featurization['Val'])

        # TODO: change this and esm encoder to gpu for release
        self.device = torch.device('cuda')
        self.model = MorganPredictor(10, 1280, fingerprint_len=fpt_len)

        if model_file is None:
            state_dict = torch.load('maspr_model.pt', map_location=torch.device('cuda'))
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
        else:
            self.model = torch.load(model_file)
            self.model = self.model.to(self.device)

        self.model.eval()

    def extract_stachelhaus(self, sequence):
        aden_predict = subprocess.run(['./aden_predict', 'stachelhaus-extract', '--query', sequence], capture_output=True)
        assert aden_predict.returncode == 0, "aden_predict failed"
        aden_resi = aden_predict.stdout.decode('utf-8').split('\n')[0].split(',')
        aden_resn = aden_predict.stdout.decode('utf-8').split('\n')[1]
        aden_resi = [int(res) for res in aden_resi]
        # return aden_resn, aden_resi
        return aden_resi

    # Should extract all sequences in the input gene
    def extract_sequence(self, sequence):
        aden_predict = subprocess.run(['./aden_predict', 'extract-sequence', '--query', sequence], capture_output=True)
        assert aden_predict.returncode == 0, "aden_predict failed"
        adom_seqs_and_pos = aden_predict.stdout.decode('utf-8').split('\n')
        adom_seqs = [seq.split(',')[0] for seq in adom_seqs_and_pos][:-1]
        return adom_seqs

    # TODO: generalize this to allow users to create their own training data
    def extract_embedding(self, sequence):
        """
        sequence: the input adomain sequence
        
        returns: the 10 x 1280 embedding of the adomain sequence's stachelhaus residues
        """
        adom_seqs = self.extract_sequence(sequence)
        embeds = []
        for adom_seq in adom_seqs:
            stach_residues = self.extract_stachelhaus(adom_seq)
            adom_embedding = self.esm_featurizer.featurize(adom_seq)
            embeds.append(adom_embedding[stach_residues])

        # return only the indices in embedding correspoding to stachelhaus residues
        embeds = torch.stack((embeds))
        print(embeds.shape)
        return embeds

    def predict(self, sequence):
        """
        sequence: the input adomain sequence

        returns: the top 5 predicted adomain classes based on the substrates in substrate_smiles.py
        """
        adom_embedding = self.extract_embedding(sequence).to(self.device)
        with torch.no_grad():
            pred_fpt, phi = self.model.forward(adom_embedding)
            sorted_outputs = fingerprint_projection(pred_fpt, self.model, device=self.device)
            top_k_labels = sorted_outputs[:, :5]
            return top_k_labels

def main():
    """
    Usage: python predict_adomain.py -i <input_adomain_sequence>
    """
    model = MASPR()
    parser = argparse.ArgumentParser(description='Predict adomain class')
    parser.add_argument('-i', '--input', type=str, help='Input adomain sequence')
    args = parser.parse_args()

    # TODO: add a mode which accepts a file or list of adomain sequences
    adom_sequence = args.input
    labels = model.predict(adom_sequence)

    # print(f'Predicted adomain classes: {labels}')
    print(f'{labels}')

if __name__ == '__main__':
    main()
