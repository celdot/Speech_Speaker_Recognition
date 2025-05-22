
# DT2119, Lab 4 End-to-end Speech Recognition

import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torchaudio.transforms import (FrequencyMasking, MelSpectrogram, Resample,
                                   TimeMasking)

# Variables to be defined --------------------------------------
''' 
train-time audio transform object, that transforms waveform -> spectrogram, with augmentation
''' 
train_audio_transform = torch.nn.Sequential(
    MelSpectrogram(sample_rate = 16000, n_mels=80),
    FrequencyMasking(freq_mask_param=15),
    TimeMasking(time_mask_param=35)
)
'''
test-time audio transform object, that transforms waveform -> spectrogram, without augmentation 
'''
test_audio_transform = torch.nn.Sequential(
    MelSpectrogram(sample_rate = 16000, n_mels=80)
)

# Functions to be implemented ----------------------------------

def intToStr(labels):
    """
        convert list of integers to string
    Args: 
        labels: list of ints
    Returns:
        string with space-separated characters
    """
    num_to_char = {0: "'", 1: "_"}
    num_to_char.update({i + 2: chr(i + 97) for i in range(26)})
    
    res = []
    for i in range(len(labels)):
        res.append(num_to_char[labels[i]])
    return ' '.join(res)


def strToInt(text):
    """
        convert string to list of integers
    Args:
        text: string
    Returns:
        list of ints
    """
    char_to_num = {"'": 0, "_": 1}
    char_to_num.update({chr(i + 97): i + 2 for i in range(26)})
    
    text = text.replace(" ", "_").lower()
    res = []
    for i in range(len(text)):
        res.append(char_to_num[text[i]])
    return res

def dataProcessing(data, transform):
    '''
    process a batch of speech data
    arguments:
        data: list of tuples, representing one batch. Each tuple is of the form
            (waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)
        transform: audio transform to apply to the waveform
    returns:
        a tuple of (spectrograms, labels, input_lengths, label_lengths) 
        -   spectrograms - tensor of shape B x C x T x M 
            where B=batch_size, C=channel, T=time_frames, M=mel_band.
            spectrograms are padded the longest length in the batch.
        -   labels - tensor of shape B x L where L is label_length. 
            labels are padded to the longest length in the batch. 
        -   input_lengths - list of half spectrogram lengths before padding
        -   label_lengths - list of label lengths before padding
    '''
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for waveform, _, utterance, _, _, _ in data:
        # Apply transform to waveform
        spec = transform(waveform)  # shape: [1, n_mels, time]
        spec = spec.squeeze(0).transpose(0, 1)  # shape: [time, n_mels]
        spectrograms.append(spec)

        # Convert utterance to int sequence
        label = torch.tensor(strToInt(utterance), dtype=torch.long)
        labels.append(label)

        # Track lengths before padding
        input_lengths.append(spec.shape[0] // 2)
        label_lengths.append(len(label))

    # Pad sequences
    spectrograms = pad_sequence(spectrograms, batch_first=True)  # shape: [B, T, M]
    spectrograms = spectrograms.unsqueeze(1).transpose(2, 3)     # shape: [B, C, M, T] # C=1

    labels = pad_sequence(labels, batch_first=True)  # shape: [B, L]

    return spectrograms, labels, input_lengths, label_lengths
    
def greedyDecoder(output, blank_label=28):
    '''
    decode a batch of utterances
    arguments:
        output: network output tensor, shape B x T x C where B=batch_size, T=time_steps, C=characters
        blank_label: id of the blank label token
    returns:
        list of decoded strings
    '''
    # Get the index of the highest probability character at each time step
    max_probs = torch.argmax(output, dim=2)  # shape: (B, T)
    
    decoded_strings = []
    
    for batch in max_probs:
        prev_char = None
        decoded = []
        
        for char_idx in batch:
            char_idx = char_idx.item()
            # Skip blank and repeated characters
            if char_idx != blank_label and char_idx != prev_char:
                decoded.append(char_idx)
            prev_char = char_idx
        
        # Convert character indices to characters
        decoded_strings.append(intToStr(decoded))
    
    return decoded_strings

def levenshteinDistance(ref,hyp):
    '''
    calculate levenshtein distance (edit distance) between two sequences
    arguments:
        ref: reference sequence
        hyp: sequence to compare against the reference
    output:
        edit distance (int)
    '''
    m = len(ref)
    n = len(hyp)

    # Initialize the distance matrix
    d = np.zeros((m + 1, n + 1), dtype=int)

    # Fill the first row and first column
    d[:, 0] = np.arange(m + 1)
    d[0, :] = np.arange(n + 1)

    # Compute Levenshtein distance
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(
                    d[i - 1, j] + 1,    # deletion
                    d[i, j - 1] + 1,    # insertion
                    d[i - 1, j - 1] + 1 # substitution
                )

    return d[m, n]
