import os
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import load_model

contiglen = 0.15 #unit in k
data_path = '/scratch1/fangyunl/DeepVirData/data_cathy/'
model = 'result_12-12_20ep_c_fixed/'
model_path = '/scratch1/fangyunl/DeepVirData/'+model


# Load the model
model = load_model(model_path+"trained_model.keras")

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
tr_data_path = data_path+'tr/encode/'
val_data_path = data_path+'val/encode/'
test_data_path = data_path+'test/encode/'
print("...loading virus data...")
phage_train = load_data(tr_data_path, 'virus', contiglen)
phage_val = load_data(val_data_path, 'virus', contiglen)
phage_test = load_data(test_data_path, 'virus', contiglen)
print(f"training virus contains {phage_train[0].shape[0]} fragments")
print(f"validation virus contains {phage_val[0].shape[0]} fragments")
print(f"testing virus contains {phage_test[0].shape[0]} fragments")

## Load host data
print("...loading host data...")
host_train = load_data(tr_data_path, 'host', contiglen)
host_val = load_data(val_data_path, 'host', contiglen)
host_test = load_data(test_data_path, 'host', contiglen)
print(f"training prokaryotes contains {host_train[0].shape[0]} fragments")
print(f"validation prokaryotes contains {host_val[0].shape[0]} fragments")
print(f"testing prokaryotes contains {host_test[0].shape[0]} fragments")

## Combine and shuffle training data
print("...combining and processing training data...")
X_trfw_shuf, X_trbw_shuf, Y_tr_shuf = combine_and_shuffle(host_train, phage_train)

## Combine and prepare validation data and test data
print("...combining and processing validation data...")
X_valfw = np.concatenate((host_val[0], phage_val[0]), axis=0)
X_valbw = np.concatenate((host_val[1], phage_val[1]), axis=0)
Y_val = np.concatenate((np.zeros(host_val[0].shape[0]), np.ones(phage_val[0].shape[0])))

print("...combining and processing test data...")
X_testfw = np.concatenate((host_test[0], phage_test[0]), axis=0)
X_testbw = np.concatenate((host_test[1], phage_test[1]), axis=0)
Y_test = np.concatenate((np.zeros(host_test[0].shape[0]), np.ones(phage_test[0].shape[0])))
del host_train, phage_train, host_val, phage_val, host_test, phage_test


def calculate_roc_and_auc(model, X_fw, X_bw, Y, batch_size):
    Y_pred = model.predict([X_fw, X_bw], batch_size=batch_size)
    fpr, tpr, _ = roc_curve(Y, Y_pred)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

batch_size = int(X_trfw_shuf.shape[0]/(1000*1000/contiglen))

# Calculate ROC and AUC for each dataset
fpr_train, tpr_train, auc_train = calculate_roc_and_auc(model, X_trfw_shuf, X_trbw_shuf, Y_tr_shuf, batch_size)
fpr_val, tpr_val, auc_val = calculate_roc_and_auc(model, X_valfw, X_valbw, Y_val, batch_size)
fpr_test, tpr_test, auc_test = calculate_roc_and_auc(model, X_testfw, X_testbw, Y_test, batch_size)

# Plot all ROC curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'Training AUROC = {auc_train:.2f}')
plt.plot(fpr_val, tpr_val, color='green', lw=2, label=f'Validation AUROC = {auc_val:.2f}')
plt.plot(fpr_test, tpr_test, color='red', lw=2, label=f'Test AUROC = {auc_test:.2f}')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic (ROC) Curve (contig length {contiglen}k)')
plt.legend(loc='lower right')
plt.grid()

# Save the plot
plot_path = f'auroc_plot_{contiglen}.png'  # Specify the output file name with contig length
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()