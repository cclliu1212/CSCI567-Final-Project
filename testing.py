import os
import random
import time
import openpyxl
import keras
from keras.models import load_model, Model
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report, roc_auc_score

contiglen = 0.3 #unit in k
data_path = '/scratch1/fangyunl/DeepVirData/data_cathy/'
model = 'result_12-12_20ep_c_0.3_fixed/'
model_path = '/scratch1/fangyunl/DeepVirData/'+model
batch_size = 32

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

# Evaluate a TensorFlow model
loss, accuracy = model.evaluate([X_fw_shuf, X_bw_shuf], Y_shuf, batch_size=batch_size)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Generate predictions
Y_pred = model.predict([X_fw_shuf, X_bw_shuf], batch_size=batch_size)
Y_pred_classes = (Y_pred > 0.5).astype(int)

# Calculate AUC
auc = roc_auc_score(Y_shuf, Y_pred)
print(f'auc={auc}\n')


# Calculate additional metrics

import pandas as pd
report = classification_report(Y_shuf, Y_pred_classes)
with open(f"test/classification_report_{contiglen}.txt", "w") as f:
    f.write(report)

