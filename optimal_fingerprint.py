import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs import FingerprintSimilarity
from rdkit.Chem import Descriptors

from substrate_smiles import sub_to_smiles

def partial_charge(label):
    """
    label: the label of the substrate

    returns: the partial charges of the substrate
    """
    mol = Chem.MolFromSmiles(sub_to_smiles[label])
    AllChem.ComputeGasteigerCharges(mol)
    gasteiger_charges = [round(float(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge')), 2) for i in range(mol.GetNumAtoms())]
    return set(gasteiger_charges)

def morgan_and_maccs_for_label(label):
    """
    label: the label of the substrate

    returns: the Morgan fingerprint and MACCS keys of the substrate
    for more information see http://rdkit.org/docs/source/rdkit.Chem.MACCSkeys.html
    and https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
    """
    mol = Chem.MolFromSmiles(sub_to_smiles[label])
    fpt1 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=128)
    fpt2 = MACCSkeys.GenMACCSKeys(mol)
    return fpt1, fpt2

# Returns a numpy array featurization for a given label.
def featurize(label):
    """
    label: the label of the substrate

    returns: a numpy array featurization of the substrate
    the returned fingerprint is a concatenation of the Morgan fingerprint and MACCS keys.
    its first 128 bits are the Morgan fingerprint, the next 166 bits are the MACCS keys,
    and the last bit is the partial charge.
    """
    fpt1, fpt2 = morgan_and_maccs_for_label(label)

    # subtract out charges which are shared with Glycine.
    sidechains = label_to_charge[label] - label_to_charge['Gly']
    charge = np.average(list(sidechains)) if len(sidechains) > 0 else 0

    # convert fpts to a list
    fpt1 = [float(c) for c in fpt1.ToBitString()]
    fpt2 = [float(c) for c in fpt2.ToBitString()]

    vec = fpt1 + fpt2 + [5 * float(charge)]
    return np.array(vec)

def cosine_distance(vec1, vec2):
    """ computes the cosine distance between two vectors """
    return 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def l2_distance(vec1, vec2):
    """ computes the l2 distance between two vectors """
    return np.linalg.norm(vec1 - vec2)

label_to_charge = {label: partial_charge(label) for label in sub_to_smiles}
label_to_featurization = {label: featurize(label) for label in sub_to_smiles}

def main():
    # tSNE visualization
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from matplotlib.cm import get_cmap
    from matplotlib.lines import Line2D
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    # Computes `feat_distance` for two labels.
    def fpt_distance(label1, label2):
        vec1 = label_to_featurization[label1]
        vec2 = label_to_featurization[label2]

        # can easily return the feature below
        return (1 - jaccard_index(vec1[:128], vec2[:128])) + (1 - jaccard_index(vec1[128:-1], vec2[128:-1])) + np.abs(vec1[-1] - vec2[-1])

    labels = list(sub_to_smiles.keys())

    # Compute distance matrix
    dist_matrix = np.zeros((len(labels), len(labels)))
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            if i < j:  # Compute only for one half as the matrix is symmetric
                dist = fpt_distance(label1, label2)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

    i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    # Convert the upper triangular distance matrix to a condensed matrix required by linkage
    condensed_dist_matrix = squareform(dist_matrix, checks=False)

    # Perform hierarchical clustering
    Z = linkage(condensed_dist_matrix, 'ward')  # 'ward' is one method of hierarchical clustering

    # Form flat clusters from the hierarchical clustering defined by the linkage matrix
    # You can choose a threshold for clustering or decide it based on a dendrogram or other criteria
    # max_d = 1.2
    # clusters = fcluster(Z, max_d, criterion='distance')
    n_clusters = 14
    clusters = fcluster(Z, n_clusters, criterion='maxclust')

    # Map labels to their clusters
    label_to_cluster = {label: cluster for label, cluster in zip(labels, clusters)}

    cluster_to_labels = {}
    for label, cluster in label_to_cluster.items():
        if cluster not in cluster_to_labels:
            cluster_to_labels[cluster] = []
        cluster_to_labels[cluster].append(label)

    for cls in cluster_to_labels:
        print(f'Cluster {cls}: {cluster_to_labels[cls]}')

    # Make a tsne plot of the label
    labels = list(label_to_featurization.keys())
    featurizations = np.array(list(label_to_featurization.values()))

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    featurizations_2d = tsne.fit_transform(featurizations)

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'darkred', 'lightgreen']
    markers = ['o', 's', '^', 'P', '*', 'X', 'd', 'v', '<', '>']

    # Ensure we have enough colors and markers for the labels
    if len(set(labels)) > len(colors) * len(markers):
        raise ValueError("Number of unique labels exceeds the number of available colors or markers.")

    # Create color and marker mapping for each label
    unique_labels = sorted(set(labels))  # Sort to ensure consistent mapping
    n_labels = len(unique_labels)
    label_to_color_marker = {label: (colors[i % len(colors)], markers[i // len(colors)]) for i, label in enumerate(unique_labels)}

    # Plotting
    plt.figure(figsize=(10, 10))
    for i, label in enumerate(labels):
        color, marker = label_to_color_marker[label]
        plt.scatter(featurizations_2d[i, 0], featurizations_2d[i, 1], color=color, marker=marker, alpha=0.7, label=label if i == labels.index(label) else "")

    # Create a legend mapping colors and markers to labels
    legend_elements = [Line2D([0], [0], marker=marker, color='w', label=label,
                              markerfacecolor=color, markersize=10)
                       for label, (color, marker) in label_to_color_marker.items()]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)

    plt.title('t-SNE of Molecule Featurizations')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.savefig('fingerprint_tsne.png')
    plt.show()

if __name__ == '__main__':
    main()







