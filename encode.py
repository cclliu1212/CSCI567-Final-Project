import os
import numpy as np
from Bio import SeqIO
import random

fragment_length = 300

input_path = '/scratch1/fangyunl/DeepVirData/data/'
output_path = '/scratch1/fangyunl/DeepVirData/data_cathy/'

def encode_sequence(sequence):
    # One-hot encode a sequence
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    return np.array([mapping.get(base, [0.25, 0.25, 0.25, 0.25]) for base in sequence.upper()])

def fragment_sequence(sequence, fragment_length):
    fragments = []
    for i in range(0, len(sequence) - fragment_length + 1, fragment_length):
        fragments.append(sequence[i:i + fragment_length])
    return fragments

def reverse_complement(sequence):
    complement = str.maketrans('ACGTacgt', 'TGCAtgca')
    return sequence.translate(complement)[::-1]

def filter_sequence(sequence):
    # Filter out sequences with more than 30% 'N'
    num_n = sequence.upper().count('N')
    if num_n / len(sequence) > 0.3:
        return False
    return True

def process_fasta(file_path, fragment_length):
    forward_sequences = []
    reverse_sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        # Fragment the sequence
        fragments = fragment_sequence(str(record.seq), fragment_length)
        for fragment in fragments:
            # Apply filtering
            if not filter_sequence(fragment):
                continue  # Skip sequences with more than 30% 'N'

            # Encode forward sequence
            forward_encoded = encode_sequence(fragment)
            forward_sequences.append(forward_encoded)

            # Generate and encode reverse complement
            reverse_encoded = encode_sequence(reverse_complement(fragment))
            reverse_sequences.append(reverse_encoded)

    return np.array(forward_sequences), np.array(reverse_sequences)

def process_fasta_with_limit(file_path, fragment_length, limit, random_seed):
    random.seed(random_seed)  # Set random seed for reproducibility
    forward_sequences = []
    reverse_sequences = []

    all_fragments = []
    for record in SeqIO.parse(file_path, "fasta"):
        # Fragment the sequence
        fragments = fragment_sequence(str(record.seq), fragment_length)
        # Filter and add valid fragments to the list
        all_fragments.extend([fragment for fragment in fragments if filter_sequence(fragment)])

    # Shuffle fragments and take a subset of size 'limit'
    random.shuffle(all_fragments)
    selected_fragments = all_fragments[:limit]

    # Process the selected fragments
    for fragment in selected_fragments:
        forward_encoded = encode_sequence(fragment)
        forward_sequences.append(forward_encoded)

        reverse_encoded = encode_sequence(reverse_complement(fragment))
        reverse_sequences.append(reverse_encoded)

    return np.array(forward_sequences), np.array(reverse_sequences)

###training
print("... encoding training dataset ...")
type = 'tr/'
viral_fasta = input_path+type+"training_virus.fasta" 
prokaryotic_fasta = input_path+type+"training_prok.fasta"

viral_forward, viral_reverse = process_fasta(viral_fasta, fragment_length)
str_frg_len = fragment_length/1000
str_frg_len = str(str_frg_len)+'k'
np.save(output_path+type+'encode/viral#'+str(str_frg_len)+'_codefw.npy', viral_forward)
np.save(output_path+type+'encode/viral#'+str(str_frg_len)+'_codebw.npy', viral_reverse)

print("Shape of viral forward sequences:", viral_forward.shape)
print("Shape of viral reverse sequences:", viral_reverse.shape)

num_viral_fragments = viral_forward.shape[0]

prokaryotic_forward, prokaryotic_reverse = process_fasta_with_limit(
    prokaryotic_fasta, fragment_length, num_viral_fragments, random_seed=42
)
np.save(output_path+type+'encode/host#'+str(str_frg_len)+'_codefw.npy', prokaryotic_forward)
np.save(output_path+type+'encode/host#'+str(str_frg_len)+'_codebw.npy', prokaryotic_reverse)

print("Shape of prokaryotic forward sequences:", prokaryotic_forward.shape)
print("Shape of prokaryotic reverse sequences:", prokaryotic_reverse.shape)


###validation
print("... encoding validation dataset ...")
type = 'val/'

viral_fasta = input_path+type+"validation_virus.fasta" 
prokaryotic_fasta = input_path+type+"validation_prok.fasta"

viral_forward, viral_reverse = process_fasta(viral_fasta, fragment_length)
np.save(output_path+type+'encode/viral#'+str(str_frg_len)+'_codefw.npy', viral_forward)
np.save(output_path+type+'encode/viral#'+str(str_frg_len)+'_codebw.npy', viral_reverse)

print("Shape of viral forward sequences:", viral_forward.shape)
print("Shape of viral reverse sequences:", viral_reverse.shape)

num_viral_fragments = viral_forward.shape[0]

prokaryotic_forward, prokaryotic_reverse = process_fasta_with_limit(
    prokaryotic_fasta, fragment_length, num_viral_fragments, random_seed=42
)
np.save(output_path+type+'encode/host#'+str(str_frg_len)+'_codefw.npy', prokaryotic_forward)
np.save(output_path+type+'encode/host#'+str(str_frg_len)+'_codebw.npy', prokaryotic_reverse)

print("Shape of prokaryotic forward sequences:", prokaryotic_forward.shape)
print("Shape of prokaryotic reverse sequences:", prokaryotic_reverse.shape)


###testing
print("... encoding testing dataset ...")
type = 'test/'
viral_fasta = input_path+type+"testing_virus.fasta" 
prokaryotic_fasta = input_path+type+"testing_prok.fasta"

viral_forward, viral_reverse = process_fasta(viral_fasta, fragment_length)
np.save(output_path+type+'encode/virus#'+str(str_frg_len)+'_codefw.npy', viral_forward)
np.save(output_path+type+'encode/virus#'+str(str_frg_len)+'_codebw.npy', viral_reverse)

print("Shape of viral forward sequences:", viral_forward.shape)
print("Shape of viral reverse sequences:", viral_reverse.shape)

num_viral_fragments = viral_forward.shape[0]

prokaryotic_forward, prokaryotic_reverse = process_fasta_with_limit(
    prokaryotic_fasta, fragment_length, num_viral_fragments, random_seed=42
)
np.save(output_path+type+'encode/host#'+str(str_frg_len)+'_codefw.npy', prokaryotic_forward)
np.save(output_path+type+'encode/host#'+str(str_frg_len)+'_codebw.npy', prokaryotic_reverse)

print("Shape of prokaryotic forward sequences:", prokaryotic_forward.shape)
print("Shape of prokaryotic reverse sequences:", prokaryotic_reverse.shape)
