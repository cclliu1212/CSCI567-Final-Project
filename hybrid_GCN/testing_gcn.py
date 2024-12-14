import os
import random
import time
import keras
from keras.models import load_model, Model
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report, roc_auc_score

contiglen = 0.15 #unit in k
data_path = '/scratch1/fangyunl/DeepVirData/data_cathy/'
model = 'result/'
model_path = '/scratch1/fangyunl/DeepVirData/GCN/'+model


def load_data(file_dir, keyword, contig_length):
    #Load forward and backward data files based on keyword and contig length.
    filename_fw = next(
        (x for x in os.listdir(file_dir)
        if 'codefw.npy' in x and keyword in x and f'{contig_length}k' in x),
        None
    )
    if filename_fw is None:
        raise FileNotFoundError(
            f"No file matching 'codefw.npy', keyword '{keyword}', and '{contig_length}k' found in {file_dir}"
        )
    print(f"Loading data: {filename_fw}")
    data_fw = np.load(os.path.join(file_dir, filename_fw))
    data_bw = np.load(os.path.join(file_dir, filename_fw.replace('fw', 'bw')))
    return data_fw, data_bw
    
def combine_and_shuffle(data_host, data_phage):
    #Combine host and phage data, create labels, and shuffle.
    labels = np.concatenate((np.zeros(data_host[0].shape[0]), np.ones(data_phage[0].shape[0])))
    data_fw = np.concatenate((data_host[0], data_phage[0]), axis=0)
    data_bw = np.concatenate((data_host[1], data_phage[1]), axis=0)
    indices = np.arange(data_fw.shape[0])
    np.random.shuffle(indices)
    
    data_fw_shuf = data_fw[indices]
    data_bw_shuf = data_bw[indices]
    labels_shuf = labels[indices]
    
    return data_fw_shuf, data_bw_shuf, labels_shuf

## Load virus data
test_data_path = data_path+'test/encode/'
print("...loading virus data...")
phage_test = load_data(test_data_path, 'virus', contiglen)

## Load host data
print("...loading prokaryotes data...")
host_test = load_data(test_data_path, 'host', contiglen)


model = load_model(model_path+"trained_model.keras")

## Combine and shuffle training data
print("...combining and processing training data...")
X_fw_shuf, X_bw_shuf, Y_shuf = combine_and_shuffle(host_test, phage_test)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Reverse encoding: Convert one-hot encoded sequences back to DNA strings
def decode_sequences(encoded_sequences):
    """
    Decode one-hot encoded sequences into nucleotide strings (A, T, C, G).
    """
    # Define the mapping from one-hot indices to nucleotides
    nucleotide_map = ['A', 'T', 'C', 'G']
    decoded_sequences = []

    for encoded_seq in encoded_sequences:
        decoded_seq = ''.join([nucleotide_map[np.argmax(base)] for base in encoded_seq])
        decoded_sequences.append(decoded_seq)

    return decoded_sequences

# Generate 3-mers from decoded sequences
def generate_3mers(sequence, k=3):
    """
    Generate overlapping k-mers from a nucleotide sequence.
    """
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]


def preprocess_gcn_inputs(encoded_sequences, num_nodes, embedding_dim, threshold=0.5):
    """
    Preprocess one-hot encoded sequences into GCN-compatible node features and adjacency matrices.
    """
    # Step 1: Decode sequences
    decoded_sequences = decode_sequences(encoded_sequences)

    # Step 2: Generate 3-mers
    all_3mers = [generate_3mers(seq) for seq in decoded_sequences]
    flattened_3mers = [' '.join(mers) for mers in all_3mers]

    # Step 3: TF-IDF for node features
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(flattened_3mers).toarray()

    # Pad TF-IDF matrix to match num_nodes
    node_features = np.zeros((len(decoded_sequences), num_nodes, embedding_dim))
    for i in range(len(decoded_sequences)):
        num_mers = min(tfidf_matrix.shape[1], num_nodes)
        node_features[i, :num_mers, :] = tfidf_matrix[i, :num_mers, None]

    # Step 4: Compute PMI adjacency matrices
    adjacency_matrices = np.zeros((len(decoded_sequences), num_nodes, num_nodes))
    for i in range(len(decoded_sequences)):
        similarity_matrix = cosine_similarity(node_features[i])
        adjacency_matrix = (similarity_matrix > threshold).astype(float)
        adjacency_matrices[i] = adjacency_matrix

    return node_features, adjacency_matrices

gcn_node_features_test, adjacency_matrices_test = preprocess_gcn_inputs(
    X_fw_shuf, num_nodes=64, embedding_dim=30, threshold=0.5
)
batch_size = int(X_fw_shuf.shape[0]/(1000*1000/contiglen))

# Evaluate a TensorFlow model
loss, accuracy = model.evaluate([gcn_node_features_test, adjacency_matrices_test, X_fw_shuf, X_bw_shuf, X_fw_shuf], Y_shuf, batch_size=batch_size)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Generate predictions
Y_pred = model.predict([gcn_node_features_test, adjacency_matrices_test, X_fw_shuf, X_bw_shuf, X_fw_shuf], batch_size=batch_size)
Y_pred_classes = (Y_pred > 0.5).astype(int)

# Calculate AUC
auc = roc_auc_score(Y_shuf, Y_pred)
print(f'auc={auc}\n')


# Calculate additional metrics
report = classification_report(Y_shuf, Y_pred_classes)
with open(f"test/classification_report_{contiglen}.txt", "w") as f:
    f.write(report)

