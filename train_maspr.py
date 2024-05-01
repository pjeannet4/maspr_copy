import argparse
import glob
import pandas as pd
import numpy as np

import torch
import torch.utils.data as torch_data
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

from optimal_fingerprint import label_to_featurization
from metrics import old_topx, morgan_topx, fingerprint_projection, hamming_accuracy, encoded_signature, get_min_hamming_distance_one_hot, hamming_bucket, n_buckets

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

def esm_for_idx(idx):
    """
    Loads the ESM embedding for a given index in the training data file.
    
    This function assumes that the training data has been processed wth the ESM-2 model.
    """
    esm = np.load(f'../adenylation-docking/adomain_seq_big/stachelhaus_{idx}.npy')
    return esm

# Load the fingerprint for a label in the training data file.
def fingerprint_for_label(label):
    """
    Loads the interpretable molecular fingerprint for a given substrate.
    """
    feats = label_to_featurization[label]
    return feats

class FingerprintClassifier(nn.Module):
    """
    A simple classifier that takes a fingerprint as input and predicts the substrate label.

    This is a classifier head that is attached to the MorganPredictor model. The outputs of
    the classifier are discarded after training, but the hidden layer is used as an A-domain-
    specific substrate embedding for nearest substrate search.
    """
    def __init__(self, fpt_len=296, classes=41):
        super(FingerprintClassifier, self).__init__()
        self.fc1 = nn.Linear(fpt_len, fpt_len)
        self.norm1 = nn.LayerNorm(fpt_len)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fpt_len, classes)

        # initialize fc1 to be the identity function, to not deviate so much from the fingerprint
        self.fc1.weight.data = torch.eye(fpt_len)

        # initialize an optimizer for this model
        self.optimizer = optim.AdamW(self.parameters(), lr=0.0001)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass for the fingerprint classifier.

        This returns the predicted label and the hidden layer embedding.
        """
        x = self.fc1(x)
        x = self.norm1(x)
        phi = self.relu(x)
        x = self.fc2(phi)
        # return the predicted label and the fingerprint
        return x, phi

    def update(self, fpt, label):
        """
        Update the fingerprint classifier with a new batch of fingerprints and labels.
        """
        self.train()
        self.optimizer.zero_grad()
        output, _ = self.forward(fpt)
        loss = self.criterion(output, label)
        loss.backward()
        self.optimizer.step()

    def embed_fpt(self, fpt):
        """
        Embed a fingerprint by returning the hidden layer embedding.
        """
        self.eval()
        with torch.no_grad():
            _, phi = self.forward(fpt)
            return phi

class MorganPredictor(nn.Module):
    def __init__(self, num_residues, esm_embed_dim, proj_dim1=480, proj_dim2=240, hidden_dim1=240, fingerprint_len=296):
        """
        Initialize the MorganPredictor model.

        `num_residues` is the number of residues in the ESM embedding.
        `esm_embed_dim` is the dimension of the ESM embedding model used. By default this is 1280.
        `proj_dim1` is the size of the first layer that projects the ESM embedding.
        `proj_dim2` is the size of the second layer that projects the ESM embedding.
        `hidden_dim1` is the size of the hidden layers after flattening the representation.
        `fingerprint_len` is the length of the fingerprint that is predicted.
        """
        super(MorganPredictor, self).__init__()

        # Shared projection across all stachelhaus indices.
        self.proj1 = nn.Linear(esm_embed_dim, proj_dim1)
        self.pre_relu1 = nn.ELU()

        self.proj2 = nn.Linear(proj_dim1, proj_dim2)
        self.pre_relu2 = nn.ELU()

        # Flatten the representation to eventually predict the fingerprint.
        self.flat = nn.Flatten()
        self.norm0 = nn.LayerNorm(proj_dim2 * num_residues)

        # Hidden layers for after flattening.
        self.linear1 = nn.Linear(proj_dim2 * num_residues, hidden_dim1)
        self.norm1 = nn.LayerNorm(hidden_dim1)
        self.relu1 = nn.ELU()

        self.linear2 = nn.Linear(hidden_dim1, hidden_dim1)
        self.norm2 = nn.LayerNorm(hidden_dim1)
        self.relu2 = nn.ELU()

        # Final layer to predict the fingerprint.
        self.linear3 = nn.Linear(hidden_dim1, fingerprint_len)

        # Initialize the fingerprint classifier which computes latent embeddings from fingerprints.
        self.fpt_classifier = FingerprintClassifier(fingerprint_len, classes=len(label_to_featurization))

    def forward(self, x, true_fpt=None, y=None):
        """
        Forward pass for the MorganPredictor model.

        This returns both the predicted fingerprints, and the embedding of those fingerprints which are to
        be used for nearest substrate search.
        """
        # x's dimension is (batch_size, 10, 1280)
        x = self.proj1(x)
        x = self.pre_relu1(x) # (batch_size, 10, 480)

        x = self.proj2(x)
        x = self.pre_relu2(x) # (batch_size, 10, 240)

        x = self.flat(x)
        x = self.norm0(x) # (batch_size, 2400)

        x = self.linear1(x)
        x = self.norm1(x)
        x = self.relu1(x) # (batch_size, 240)

        x = self.linear2(x)
        x = self.norm2(x)
        x = self.relu2(x) # (batch_size, 240)

        # this predicts the fingerprint
        fpt = self.linear3(x) # (batch_size, 296)
        pred_fpt = fpt.detach()

        # if this model is in training mode, train the inner model to predict the class from the fingerprint
        if self.training:
            assert y is not None
            self.fpt_classifier.update(pred_fpt, y)
            self.fpt_classifier.update(true_fpt, y)

        # detach and copy the fingerprint to prevent backpropagation
        fpt_copy = fpt.detach()
        phi = self.fpt_classifier.embed_fpt(fpt_copy)

        return fpt, phi

    def embed_fpt(self, fpt):
        """
        Project a fingerprint into the latent space of the inner classifier head.
        """
        projs = self.fpt_classifier.embed_fpt(fpt)
        return projs

def train_step(model, train_dataloader, criterion, optimizer, device='cuda'):
    """
    Perform a single training step for the model.
    """

    model.train()
    train_loss = 0; total = 0
    for esm_feats, label, fpt in train_dataloader:
        esm_feats, label, fpt = esm_feats.to(device), label.to(device), fpt.to(device)
        batch_num = esm_feats.size(0)
        optimizer.zero_grad()
        # pass in true fingerprint and label to update MASPR's classifier head
        pred_fpt, phi = model(esm_feats, true_fpt=fpt, y=label)
        # average cosine distance across batch
        loss = (1 - criterion(pred_fpt, fpt)).mean()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_num
    train_loss /= len(train_dataloader.dataset)
    return train_loss

def eval_step(model, val_dataloader, criterion, device='cuda'):
    """
    Perform a single evaluation step for the model.

    Returns the validation accuracy and the top-1 accuracy of the model across different hamming buckets.
    """
    model.eval()
    val_loss = 0; total = 0
    with torch.no_grad():
        top1_buckets = np.zeros(n_buckets)

        for inputs, labels_top, y_fpt, buckets in val_dataloader:
            inputs, labels_top, y_fpt = inputs.to(device), labels_top.to(device), y_fpt.to(device)
            batch_num = inputs.size(0)
            total += batch_num
            
            # compute outputs and metrics
            pred_fpt, phi = model(inputs)
            loss = (1 - criterion(pred_fpt, y_fpt)).mean()
            val_loss += loss.item() * batch_num

            # Compute the fingerprint projection and rank predictions against a test database.
            sorted_outputs, truth_labels = fingerprint_projection(pred_fpt, model, truths=y_fpt)

            # Take the top sorted predictions to compare them to the truth labels.
            sorted_tops = sorted_outputs[:, 0]
            top1_buckets += batch_num * hamming_accuracy(sorted_tops, truth_labels, buckets)
            
        val_loss /= len(val_dataloader.dataset)
        top1_buckets *= (1 / total)
    
    return val_loss, top1_buckets



def main(args):
    data = pd.read_csv('adomain_context_training_data.tsv', sep='\t')

    # Load the ESM embeddings for each index in the training data.
    data['esm'] = data.index.map(esm_for_idx)
    data['label_feats'] = data['substrate'].map(fingerprint_for_label)
    data['one_hot_signature'] = data['signature'].apply(encoded_signature)

    substrate_index = {substrate: idx for idx, substrate in enumerate(data['substrate'].unique())}
    index_substrate = {idx: substrate for substrate, idx in substrate_index.items()}
    data['substrate_index'] = data['substrate'].map(substrate_index)

	# Keep track of stats across all random states.
	# Top-1 accuracy across all hamming buckets.
    all_old_top_buckets = []
    all_new_top_buckets = []

	# The top-x accuracy across all random splits.
    all_new_topx = []
    all_old_topx = []

	# Split the data into training and validation sets according to the task
    all_train_data, all_val_data = [], []
    if args.task == 'bacfung':
        all_train_data.append(data[data['domain'] != 'Eukaryota'])
        all_val_data.append(data[data['domain'] == 'Eukaryota'])
    elif args.task == 'substrate':
        states = list(data['substrate_index'].unique())
        for substrate_idx in states:
            train_data = data[data['substrate_index'] != substrate_idx]
            val_data = data[data['substrate_index'] == substrate_idx]
            all_train_data.append(train_data)
            all_val_data.append(val_data)
    elif args.task == 'ttsplit':
        for state in [1, 4, 23, 3, 77, 333, 1707, 8091997, 314159, 77077, 999, 63836, 142857]:
            train_data, val_data = train_test_split(data, test_size=0.2, random_state=state)
            all_train_data.append(train_data)
            all_val_data.append(val_data)
    else:
        assert args.task == 'train'
        all_train_data.append(data)
        all_val_data.append(data)

    for train_data, val_data in zip(all_train_data, all_val_data):
        # Extract training data for old method.
        old_X_train = np.stack(train_data['one_hot_signature'].values)

        # Extract training data for new method.
        esm_train = np.stack(train_data['esm'].values)
        X_train = esm_train.reshape((esm_train.shape[0], esm_train.shape[1], esm_train.shape[2]))

        # Extract training targets for old method.
        y_train_old = train_data['substrate'].values
        y_train_new = train_data['substrate_index'].values

        # Extract training fingerprint targets for new method.
        y_fpt_train = np.stack(train_data['label_feats'].values)
        print(X_train.shape, y_fpt_train.shape)

        # Extract validation data for old method.
        old_X_val = np.stack(val_data['one_hot_signature'].values)

        # Extract validation data for new method.
        esm_val = np.stack(val_data['esm'].values)
        X_val = esm_val.reshape((esm_val.shape[0], esm_val.shape[1], esm_val.shape[2]))

        # Extract validation targets for old method.
        y_val_old = val_data['substrate'].values
        y_val_new = val_data['substrate_index'].values

        # Extract validation fingerprint targets for new method.
        y_fpt_val = np.stack(val_data['label_feats'].values)

        # Calculate hamming bucket distance of each test point from the training set.
        if args.task == 'train':
            ohe_hamming_dist = [0] * len(old_X_val)
        else:
            ohe_hamming_dist = [get_min_hamming_distance_one_hot(old_X_train, ohe) for ohe in old_X_val]
        ohe_hamming_buckets = [hamming_bucket(dist) for dist in ohe_hamming_dist]

        # Compute metrics for the old state of the art model, AdenPredictor.
        total = len(old_X_val)
        aden_predictor= ExtraTreesClassifier(n_estimators=100, random_state=0, criterion='gini', class_weight='balanced')
        aden_predictor.fit(old_X_train, y_train_old)
        preds = aden_predictor.predict(old_X_val)
        correct = (preds == y_val_old).sum().item()
        old_aden_score_top = 100 * correct / total
        old_aden_top_hamming_buckets = hamming_accuracy(preds, y_val_old, ohe_hamming_buckets)
        all_old_top_buckets.append(old_aden_top_hamming_buckets)
        print(f"AdenPredictor Top-1: Accuracy {old_aden_score_top:.3f}, Bucket Accuracy: {old_aden_top_hamming_buckets}")

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_new, dtype=torch.long)
        y_fpt_train_tensor = torch.tensor(y_fpt_train, dtype=torch.float32)

        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_top_tensor = torch.tensor(y_val_new, dtype=torch.long)
        y_fpt_val_tensor = torch.tensor(y_fpt_val, dtype=torch.float32)

        ohe_hamming_tensor = torch.tensor(ohe_hamming_buckets, dtype=torch.int32)

        # Create data loaders
        batch_size = 128
        train_dataset = torch_data.TensorDataset(X_train_tensor, y_train_tensor, y_fpt_train_tensor)
        train_loader = torch_data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Also pass in the hamming buckets for validation.
        val_dataset = torch_data.TensorDataset(X_val_tensor, y_val_top_tensor, y_fpt_val_tensor, ohe_hamming_tensor)
        val_loader = torch_data.DataLoader(val_dataset, batch_size=1024*4, shuffle=False)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Model parameters
        num_residues = X_train.shape[1]
        esm_embed_dim = X_train.shape[2]
        model = MorganPredictor(num_residues, esm_embed_dim, fingerprint_len=y_fpt_train.shape[1]).to(device)

        # Hyperparameters
        epochs = 80
        criterion = nn.CosineSimilarity(dim=1, eps=1e-6)
        optimizer = optim.AdamW(model.parameters(), lr=0.0001)

        # Exponential decay for learning rate across epochs.
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

        # Train the model
        for epoch in range(epochs):
            curr_lr = optimizer.param_groups[0]['lr']

            # Training step
            train_loss = train_step(model, train_loader, criterion, optimizer, device=device)

            # Evaluation step
            model.eval()
            val_loss, new_top_buckets = eval_step(model, val_loader, criterion, device=device)

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {curr_lr}, Val Acc: {repr(new_top_buckets)}")
            scheduler.step()

        all_new_top_buckets.append(new_top_buckets)

        # Compute top-x metrics
        model.eval()
        for inputs, labels_top, y_fpt, buckets in val_loader:
            inputs, labels_top, y_fpt = inputs.to(device), labels_top.to(device), y_fpt.to(device)
            outputs_top, phi = model(inputs)
            sorted_outputs, truth_labels = fingerprint_projection(outputs_top, model, truths=y_fpt)

            # Calculate top-x accuracies.
            all_new_topx.append(morgan_topx(sorted_outputs, truth_labels))
            all_old_topx.append(old_topx(aden_predictor, old_X_val, y_val_old, ohe_hamming_buckets, index_substrate))

    # Print the average top-x accuracy of AdenPredictor vs MASPR.
    print()
    print(f'AdenPredictor topx: {repr(np.average(np.array(all_old_topx), axis=0))}')
    print(f'MASPR topx: {repr(np.average(np.array(all_new_topx), axis=0))}')

    # Print the average top-1 accuracy of AdenPredictor vs MASPR.
    print()
    print(f"AdenPredictor Top-1 Buckets: {repr(np.average(np.array(all_old_top_buckets), axis=0))}")
    print(f"MASPR Top-1 Buckets: {repr(np.average(np.array(all_new_top_buckets), axis=0))}")

    if args.task == 'train':
        # Save the model
        torch.save(model.state_dict(), args.model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a MASPR model.')
    parser.add_argument('--task', action='store', help='Choose the task to run. ', choices=['bacfung', 'ttsplit', 'substrate', 'train'], required=True)
    # add a path to save the model, defualt being maspr_model.pt
    parser.add_argument('--model_path', action='store', help='Path to save the model.', default='maspr_model.pt')
    args = parser.parse_args()
    main(args)
