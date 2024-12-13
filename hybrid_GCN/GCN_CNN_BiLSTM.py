# GCN + CNN + BiLSTM
## Data Preprocess
from Bio import Entrez, SeqIO
import random
import pandas as pd
import numpy as np
import time
import openpyxl
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, callbacks, optimizers, Model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
#import seaborn as sns
#import networkx as nx
#from sklearn.decomposition import PCA

ep = 20
contiglen = 0.15 #unit in k
data_path = '/scratch1/fangyunl/DeepVirData/data_cathy/'
output_path = "/scratch1/fangyunl/DeepVirData/GCN/result/"

learning_rate = 0.001
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

tr_data_path = data_path+'tr/encode/'
val_data_path = data_path+'val/encode/'
print("...loading virus data...")
phage_train = load_data(tr_data_path, 'virus', contiglen)
phage_val = load_data(val_data_path, 'virus', contiglen)
print(f"virus contains {phage_train[0].shape[0]} fragments")

## Load host data
print("...loading host data...")
host_train = load_data(tr_data_path, 'host', contiglen)
host_val = load_data(val_data_path, 'host', contiglen)
print(f"prokaryotes contains {host_train[0].shape[0]} fragments")

## Combine and shuffle training data
print("...combining and processing training data...")
X_trfw_shuf, X_trbw_shuf, Y_tr_shuf = combine_and_shuffle(host_train, phage_train)
print(f"training contains {X_trfw_shuf.shape[0]} fragments")

## Combine and prepare validation data
print("...combining and processing validation data...")
X_valfw = np.concatenate((host_val[0], phage_val[0]), axis=0)
X_valbw = np.concatenate((host_val[1], phage_val[1]), axis=0)
Y_val = np.concatenate((np.zeros(host_val[0].shape[0]), np.ones(phage_val[0].shape[0])))
print(f"validation contains {X_valfw.shape[0]} fragments")
del host_train, phage_train, host_val, phage_val

## smaller batch size can reduce memory
batch_size = int(X_trfw_shuf.shape[0]/(1000*1000/contiglen))

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


# random_seed = 42
# np.random.seed(random_seed)
# tmp_int = 5000

# train_indices = np.random.choice(X_trfw_shuf.shape[0], size=tmp_int, replace=False)
# val_indices = np.random.choice(X_valfw.shape[0], size=tmp_int, replace=False)

# X_trfw_sample = X_trfw_shuf[train_indices]
# X_trbw_sample = X_trbw_shuf[train_indices]
# Y_tr_sample = Y_tr_shuf[train_indices]

# X_valfw_sample = X_valfw[val_indices]
# X_valbw_sample = X_valbw[val_indices]
# Y_val_sample = Y_val[val_indices]

gcn_node_features_train, adjacency_matrices_train = preprocess_gcn_inputs(
    X_trfw_shuf, num_nodes=64, embedding_dim=30, threshold=0.5
)
gcn_node_features_val, adjacency_matrices_val = preprocess_gcn_inputs(
    X_valfw, num_nodes=64, embedding_dim=30, threshold=0.5
)


# Select one instance's node features
node_features = gcn_node_features_train[0]

'''
# Plot heatmap of node features
plt.figure(figsize=(4, 3))
sns.heatmap(node_features, cmap='viridis', annot=False)
plt.title("Node Features Heatmap")
plt.xlabel("Feature Dimension")
plt.ylabel("Nodes")
plt.show()
'''
# Select one instance's adjacency matrix
adj_matrix = adjacency_matrices_train[0]

'''
# Plot heatmap of the adjacency matrix
plt.figure(figsize=(3, 3))
sns.heatmap(adj_matrix, cmap='coolwarm', square=True, annot=False)
plt.title("Adjacency Matrix")
plt.xlabel("Nodes")
plt.ylabel("Nodes")
plt.show()
'''

'''
# Convert adjacency matrix to a graph
graph = nx.from_numpy_array(adj_matrix)

# Plot the graph
plt.figure(figsize=(3,3))
nx.draw(graph, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray', font_size=10)
plt.title("Graph Representation")
plt.save()
'''

'''
# Reduce dimensions with PCA
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(transformed_features)

# Scatter plot of reduced features
plt.figure(figsize=(3, 3))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c='blue', s=50, cmap='viridis')
plt.title("GCN Transformed Features (2D PCA)")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.savefig('pca.png', dpi=300, bbox_inches='tight')
plt.show()
'''

## Training Model
def build_gcn_improvedcnn_bilstm_model(input_shape, gcn_input_shape):
    """
    Build a model combining GCN for embedding, CNN for feature extraction, and BiLSTM for sequential relationships.
    The GCN uses meaningful adjacency matrices and node features.
    """
    # GCN Inputs
    gcn_input = layers.Input(shape=gcn_input_shape, name='GCN_Node_Features')  # Node features (e.g., TF-IDF)
    adjacency_matrix = layers.Input(shape=(gcn_input_shape[0], gcn_input_shape[0]), name='GCN_Adjacency_Matrix')  # Adjacency matrix

    # GCN Layers
    gcn_hidden = layers.Dense(128, activation="relu")(gcn_input)  # Initial feature transformation
    gcn_hidden = layers.Dot(axes=1)([adjacency_matrix, gcn_hidden])  # Graph convolution (adjacency matrix * features)
    gcn_output = layers.GlobalAveragePooling1D()(gcn_hidden)  # Aggregating node features into a graph embedding

    # Forward and Reverse Inputs
    fw_input = layers.Input(shape=input_shape, name='Forward_Input')  # Forward sequence
    rv_input = layers.Input(shape=input_shape, name='Reverse_Input')  # Reverse sequence

    # CNN Path (shared for forward and reverse inputs)
    def cnn_path(input_layer):
        x = layers.Conv1D(filters=500, kernel_size=10, activation='relu')(input_layer)
        # x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(500, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        return x

    fw_output = cnn_path(fw_input)  # Forward CNN path
    rv_output = cnn_path(rv_input)  # Reverse CNN path

    # Combine forward and reverse CNN outputs
    cnn_combined_output = layers.Average()([fw_output, rv_output])

    # BiLSTM Path
    bilstm_input = layers.Input(shape=input_shape, name='BiLSTM_Input')
    bilstm_output = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(bilstm_input)
    bilstm_output = layers.Dropout(0.1)(bilstm_output)

    # Combine GCN, CNN, and BiLSTM outputs
    combined_output = layers.Concatenate()([gcn_output, cnn_combined_output, bilstm_output])
    combined_output = layers.Dense(256, activation='relu')(combined_output)
    final_output = layers.Dense(1, activation='sigmoid')(combined_output)  # Binary classification output

    # Model definition
    model = Model(
        inputs=[gcn_input, adjacency_matrix, fw_input, rv_input, bilstm_input],
        outputs=final_output
    )

    return model

# Instantiate and compile the model
# batch_size = int(X_trfw_shuf.shape[0]/(1000*1000/contiglen))
batch_size = 500
input_shape = (X_trfw_shuf.shape[1], X_trfw_shuf.shape[2]) 
gcn_input_shape = (64, 30)  # Example: 64 nodes with 30-dimensional node features

# Build the model
model = build_gcn_improvedcnn_bilstm_model(input_shape, gcn_input_shape)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()
model_name = f"{output_path}trained_model.keras"
cp = callbacks.ModelCheckpoint(filepath=model_name, verbose=1,save_best_only=True)
es = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.0001, patience=5, verbose=1)

# print(gcn_node_features_train.shape)  # Should match `gcn_input_shape`
# print(adjacency_matrices_train.shape) # Should match `(batch_size, num_nodes, num_nodes)`
# print(X_trfw_sample.shape)            # Should match `input_shape`
# print(X_trbw_sample.shape)            # Should match `input_shape`
# print(Y_tr_sample.shape)              # Should match `(batch_size,)`

history = model.fit(
    x=[
    gcn_node_features_train,  # Matches `GCN_Node_Features`
    adjacency_matrices_train, # Matches `GCN_Adjacency_Matrix`
    X_trfw_shuf,            # Matches `Forward_Input`
    X_trbw_shuf,            # Matches `Reverse_Input`
    X_trfw_shuf             # Matches `BiLSTM_Input` (assuming same as forward input)
],
    y=Y_tr_shuf,
    validation_data=(
    [
        gcn_node_features_val,  # Matches `GCN_Node_Features`
        adjacency_matrices_val, # Matches `GCN_Adjacency_Matrix`
        X_valfw,         # Matches `Forward_Input`
        X_valbw,         # Matches `Reverse_Input`
        X_valfw          # Matches `BiLSTM_Input` (assuming same as forward input)
    ],
    Y_val
),
    epochs=20,
    batch_size=batch_size,
    verbose=1,
    callbacks=[cp, es]
)

### Evaluation
def predict_and_evaluate(
    model, GCN_Node_Features, GCN_Adjacency_Matrix, X_fw, X_bw, Y, data_type, out_dir, batch_size=batch_size
):
    """
    Predict and evaluate the model for the given dataset.

    Args:
        model: Trained model for prediction.
        GCN_Node_Features: Input node features for the GCN.
        GCN_Adjacency_Matrix: Input adjacency matrices for the GCN.
        X_fw: Forward input data.
        X_bw: Reverse input data.
        Y: True labels.
        data_type: 'tr' or 'val' indicating training or validation dataset.
        out_dir: Directory to save output files.
        batch_size: Batch size for predictions.
    """
    # Predict the labels
    Y_pred = model.predict(
        [GCN_Node_Features, GCN_Adjacency_Matrix, X_fw, X_bw, X_fw],  # Provide all five inputs
        batch_size=batch_size,
    )

    # Calculate AUC
    auc = metrics.roc_auc_score(Y, Y_pred)
    print(f'auc_{data_type}={auc}\n')

    '''
    # Save predictions and true labels
    np.savetxt(os.path.join(out_dir, f"{data_type}_Y_pred.txt"), np.transpose(Y_pred))
    np.savetxt(os.path.join(out_dir, f"{data_type}_Y_true.txt"), np.transpose(Y))
    '''

    return auc

# Training data
type_tr = 'tr'
auc_train = predict_and_evaluate(
    model=model,
    GCN_Node_Features=gcn_node_features_train,
    GCN_Adjacency_Matrix=adjacency_matrices_train,
    X_fw=X_trfw_shuf,
    X_bw=X_trbw_shuf,
    Y=Y_tr_shuf,
    data_type=type_tr,
    out_dir=output_path,
)

# Validation data
type_val = 'val'
auc_val = predict_and_evaluate(
    model=model,
    GCN_Node_Features=gcn_node_features_val,
    GCN_Adjacency_Matrix=adjacency_matrices_val,
    X_fw=X_valfw,
    X_bw=X_valbw,
    Y=Y_val,
    data_type=type_val,
    out_dir=output_path,
)

print(f"Train AUC: {auc_train}")
print(f"Validation AUC: {auc_val}")
