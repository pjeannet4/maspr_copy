## MASPR
Code for Modeling A-Domain Specificity using Protein Language Models paper.

This repository is under active development. Some major TODOs include allowing users to generate their own training data for new A-domains, batch processing at inference time, and [RDKit visualization](https://www.rdkit.org/docs/GettingStartedInPython.html#generating-similarity-maps-using-fingerprints) of the interpretable specificity predictions.

## Installing Dependencies
To use MASPR, you need to have [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) installed. Once you have conda installed, you will need the following packages:

```
conda create --name maspr 
conda activate maspr
conda install pip
pip install -r requirements.txt
```

## Benchmarking a MASPR model
To benchmark a MASPR model, you will first need to download the ESM embeddings for the training data (or generate them yourself). You can download these embeddings [here](https://drive.google.com/file/d/1-7iBeYCKXUepromJusNSojKGdOf8qLLA/view?usp=sharing).

To reproduce the numbers in the paper:

```
python train_maspr.py --task ttsplit
```

To reproduce the generalization benchmark (train on bacteria and test on fungi):

```
python train_maspr.py --task bacfung
```

To reproduce the zero-shot learning benchmark (leave-one-substrate-out cross-validation):

```
python train_maspr.py --task substrate
```

To train a MASPR model using all the data:

```
python train_maspr.py --task train --model_path <MODEL_PATH>
```

## Making predictions with MASPR
To predict the specificity for all A-domains in a given gene sequence:
```
python process_adomain.py -i <GENE_SEQUENCE>
```

## Zero-shot prediction with MASPR
MASPR can consider novel substrates during inference even if they were not in its training data. To enable this feature, add your desired substrates to the `sub_to_smiles` dictionary in `substrate_smiles.py`.
