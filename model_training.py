import random
import pandas as pd
import numpy as np
import time
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn import metrics

ep = 20
contiglen = 0.15 #unit in k
data_path = '/scratch1/fangyunl/DeepVirData/data/'
output_path = "/scratch1/fangyunl/DeepVirData/result/"
learning_rate = 0.001

# Define the model architecture
def build_model(input_shape):
    model = models.Sequential()
    fw_input = layers.Input(shape=input_shape, name='Forward_Input')
    rv_input = layers.Input(shape=input_shape, name='Reverse_Input')
    
    hidden_layers = [
    # First Convolutional Layer
    layers.Conv1D(filters=500, kernel_size=10, activation='relu', input_shape=input_shape),
    layers.GlobalMaxPooling1D(),

    # Dense layers
    layers.Dense(500, activation='relu'),
    layers.Dropout(0.1),  # Dropout for regularization

    # Output layer for binary classification
    layers.Dense(1, activation='sigmoid')
    ]
    
    def apply_layers(input_layer):
        x = input_layer
        for layer in hidden_layers:
            x = layer(x)
        return x
    
    fw_out = apply_layers(fw_input)
    rv_out = apply_layers(rv_input)
    combined_output = layers.Average()([fw_out, rv_out])
    return models.Model(inputs=[fw_input, rv_input], outputs=combined_output)
   


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
print(f"training virus contains {phage_train[0].shape[0]} fragments")

## Load host data
print("...loading host data...")
host_train = load_data(tr_data_path, 'host', contiglen)
host_val = load_data(val_data_path, 'host', contiglen)
print(f" training prokaryotes contains {host_train[0].shape[0]} fragments")

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

# Instantiate and compile the model
input_shape = (X_trfw_shuf.shape[1], X_trfw_shuf.shape[2])  # (fragment_length, 4) from preprocessing
model = build_model(input_shape)
optimizer = optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Display the model architecture
model.summary()
model_name = f"{output_path}trained_model.keras"
cp = callbacks.ModelCheckpoint(filepath=model_name, verbose=1,save_best_only=True)
es = callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0.0001, patience=5, verbose=1)

# Train the model
history = model.fit(
    x=[X_trfw_shuf, X_trbw_shuf], 
    y=Y_tr_shuf,
    epochs=ep,             # Number of epochs; adjust based on dataset size and performance
    batch_size=batch_size,         # Adjust batch size based on available memory
    validation_data=([X_valfw, X_valbw], Y_val),
    verbose=2,
    callbacks=[cp, es]
)

def predict_and_evaluate(model, X_fw, X_bw, Y, data_type, out_dir, batch_size=1):
    """
    Predict and evaluate the model for the given dataset.
    
    Args:
        model: Trained model for prediction.
        X_fw: Forward input data.
        X_bw: Backward input data.
        Y: True labels.
        data_type: 'tr' or 'val' indicating training or validation dataset.
        out_dir: Directory to save output files.
        batch_size: Batch size for predictions.
    """
    
    # Predict the labels
    Y_pred = model.predict([X_fw, X_bw], batch_size=batch_size)
    
    # Calculate AUC
    auc = metrics.roc_auc_score(Y, Y_pred)
    print(f'auc_{data_type}={auc}\n')
    

    # Save predictions and true labels
    np.savetxt(os.path.join(output_path, f"{data_type}fw_Y_pred.txt"), np.transpose(Y_pred))
    np.savetxt(os.path.join(output_path, f"{data_type}fw_Y_true.txt"), np.transpose(Y))

    
    return auc

type_tr = 'tr'
auc_train = predict_and_evaluate(
    model=model,
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
    X_fw=X_valfw,
    X_bw=X_valbw,
    Y=Y_val,
    data_type=type_val,
    out_dir=output_path,
)

print(f"Train AUC: {auc_train}")
print(f"Validation AUC: {auc_val}")


# Define paths to save the model and history
history_path_csv = f"{output_path}training_history.csv"
history_path_npz = f"{output_path}training_history.npz"

# Save the model (replaced with checkPointer)
#model.save(model_path)
print(f"Model saved to {model_name}")

# Save the history as a CSV
history_df = pd.DataFrame(history.history)
history_df.to_csv(history_path_csv, index=False)
print(f"Training history saved to {history_path_csv}")

# Alternatively, save history as a compressed .npz file
np.savez_compressed(history_path_npz, **history.history)
print(f"Training history saved to {history_path_npz}")