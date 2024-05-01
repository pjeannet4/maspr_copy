import pandas as pd
import numpy as np
import torch
import torch.utils.data as torch_data
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm

from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.DataStructs import FingerprintSimilarity
from rdkit.Chem import AllChem

from optimal_fingerprint import label_to_featurization, l2_distance, cosine_distance
from substrate_smiles import sub_to_smiles

all_labels = np.array(list(label_to_featurization.keys()))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
label_to_tensor = {k: torch.tensor(v).float().to(device) for k, v in label_to_featurization.items()}

# number of unique labels which appear in the training data
n_labels = len(all_labels)

n_buckets = 5

def hamming_bucket(dist):
    """
    dist: the distance from the training data

    returns: the hamming bucket of the distance
    """
    if 12 <= dist:
        return 4
    elif 9 <= dist:
        return 3
    elif 6 <= dist:
        return 2
    elif 3 <= dist:
        return 1
    return 0

# Compute the hamming distance between two ADomainSignatures.
#
# This is the distance between `signature`s in `adomain_context_training_data.tsv`.
def hamming_distance_one_hot(sig1, sig2):
    """
    sig1: the first one-hot encoded signature
    sig2: the second one-hot encoded signature

    returns: the hamming distance between the two signatures
    """
    # compute the sum
    sum_sig = sig1[:714] + sig2[:714]
    # if the sum is 2, then the two signatures are the same at that position. 
    # this gives us a similarity metric. we get hamming distance by subtracting this from the total length
    return 34 - np.sum(sum_sig == 2)

def get_min_hamming_distance_one_hot(training_sigs, test_sig):
    """
    training_sigs: the training signatures
    test_sig: the test signature

    returns: the minimum hamming distance of the test signature from the training data
    """
    return np.min([hamming_distance_one_hot(training_sig, test_sig) for training_sig in training_sigs])

def hamming_accuracy(preds, yval, buckets):
    """
    preds: the predicted labels
    yval: the true labels
    buckets: the hamming buckets of the validation data to the training data

    returns: the accuracy of the model stratified by hamming buckets
    hamming bucket i contains all the test points with hamming distance >= i to any training point
    """
    bucket_dict = {k: [] for k in range(n_buckets)}
    if torch.is_tensor(buckets):
        buckets = buckets.numpy()
    if torch.is_tensor(preds):
        preds = preds.numpy()
    if torch.is_tensor(yval):
        yval = yval.numpy()

    for p, c, h_max in zip(preds, yval, buckets):
        # a point in bucket i is in buckets < i as well
        for h in range(0, int(h_max) + 1):
            bucket_dict[h].append(p == c)

    for h in bucket_dict:
        bucket_dict[h] = np.array(bucket_dict[h])

    def score(bucket_dict):
        if len(bucket_dict) > 0:
            return bucket_dict.sum() / len(bucket_dict)
        else:
            return 0

    return np.array([score(bucket_dict[i]) for i in range(n_buckets)])

amino_acids = 'ACDEFGHIKLMNPQRSTVWY-'
aa_index = {aa: idx for idx, aa in enumerate(amino_acids)}

def encoded_signature(signature):
    """
    signature: the signature to encode

    converts the amino acid 8-angstrom signature into a one-hot encoded vector
    """
    encoded = np.zeros((34,21), dtype=np.float32)
    for i, aa in enumerate(signature):
        encoded[i, aa_index[aa]] = 1
    return encoded.flatten()

def fingerprint_projection(preds, projector, truths=None, device='cuda'):
    """
    preds: the predicted fingerprints from the maspr model's fingerprint predictor
    projector: the projector to use to project the fingerprints. this is the classifier head of the maspr model
    truths: the true fingerprints. if not provided, this function will only return the sorted predictions

    returns: sorted predictions, and ground truths substrate labels.
    the substrates considered in ranking are the ones in label_to_featurization
    """
    label_feats = torch.tensor([label_to_featurization[l] for l in all_labels]).float().to(device)
    label_projs = projector.embed_fpt(label_feats)
    label_to_proj = {l: p.detach().cpu().numpy() for l, p in zip(all_labels, label_projs)}
    pred_projs = projector.embed_fpt(preds).detach().cpu().numpy()

    if truths is not None:
        truth_projs = projector.embed_fpt(truths).detach().cpu().numpy()
    else:
        truth_projs = [None] * len(pred_projs)

    pred_labels = []
    truth_labels = []
    old_preds = preds.detach().cpu().numpy()
    num_affected = 0
    for old_pred, pred, truth in zip(old_preds, pred_projs, truth_projs):
        dists = np.array([cosine_distance(pred, label_to_proj[l]) for l in all_labels])
        sorted_inds = np.argsort(dists)
        sorted_labels = all_labels[sorted_inds]
        pred_labels.append(sorted_labels)

        if truth is not None:
            truth_dists = np.array([l2_distance(truth, label_to_proj[l]) for l in all_labels])
            # Should be zero but rounding sucks.
            assert(np.min(truth_dists) < 0.1)

            truth_label = all_labels[np.argmin(truth_dists)]
            truth_labels.append(truth_label)

    if truths is not None:
        return np.array(pred_labels), np.array(truth_labels)
    else:
        return np.array(pred_labels)

distance_bins = np.arange(0.0, 5.0, 0.25)

def morgan_topx(sorted_preds, true_labels):
    """
    sorted_preds: the sorted predictions of the maspr model
    true_labels: the ground truth labels

    returns: the topx accuracy of the method. topx[i] is 1 if ground truth is in topx[0..=i] and 0 otherwise
    """
    topx = np.zeros(n_labels)

    # for each index, calculate the position of the correct label
    for o, l in zip(sorted_preds, true_labels):
        # sort the output indices based on position
        sorted_ranks = list(o)

        # find the position of the true label in the sorted ranks
        ind_pos = sorted_ranks.index(l)
        topx[ind_pos:] += 1

    return topx / len(true_labels)

def old_topx(model, inputs, labels, buckets, ind_to_sub):
    """
    model: the model to use for prediction
    inputs: the validation data
    labels: the ground truth labels
    buckets: the hamming buckets of the validation data to the training data
    ind_to_sub: the index to substrate mapping

    returns: the topx accuracy of the method. topx[i] is 1 if ground truth is in topx[0..=i] and 0 otherwise
    """
    outputs = model.predict_proba(inputs)

    misprediction_ranks = []

    topx = np.zeros(n_labels)
    classes = model.classes_

    all_aa = set(ind_to_sub.values())

    # for each index, calculate the position of the correct label
    for i, o, l, b in zip(inputs, outputs, labels, buckets):
        # sort the output indices based on position
        pred_inds = o.argsort()
        sorted_ranks = list(model.classes_[pred_inds[::-1]])

        # find the position of the true label in the sorted ranks
        ind_pos = sorted_ranks.index(l) if l in sorted_ranks else 40
        topx[ind_pos:] += 1

    return topx / len(labels)

