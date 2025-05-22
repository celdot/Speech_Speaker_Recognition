import torch
from torchaudio.transforms import (FrequencyMasking, MelSpectrogram, Resample,
                                   TimeMasking)
from torchvision.transforms import Compose

# DT2119, Lab 4 End-to-end Speech Recognition

# Variables to be defined --------------------------------------
""" 
train-time audio transform object, that transforms waveform -> spectrogram, with augmentation
"""
train_audio_transform = Compose([
    MelSpectrogram(n_mels=80),
    FrequencyMasking(freq_mask_param=15),
    TimeMasking(time_mask_param=35),
])
"""
test-time audio transform object, that transforms waveform -> spectrogram, without augmentation 
"""
test_audio_transform = Compose([
    MelSpectrogram(n_mels=80),
])

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
    """
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
    """
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for i in range(len(data)):
        waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id = data[i]
        # apply transform to waveform
        spectrogram = transform(waveform).squeeze(0).transpose(0, 1)
        # Shape of spectrogram is (time, n_mels)
        spectrograms.append(spectrogram)

        # convert utterance to list of integers
        label = strToInt(utterance)
        label = torch.tensor(label, dtype=torch.long)
        # append lengths
        input_lengths.append(spectrogram.shape[0] // 2)
        label_lengths.append(len(label))
        labels.append(label)

    max_input_length = max([s.shape[0] for s in spectrograms])
    max_label_length = max(label_lengths)
    for i in range(len(spectrograms)):
        if spectrograms[i].shape[0] < max_input_length:
            # pad spectrograms to max_input_length
            spectrograms[i] = torch.nn.functional.pad(
                spectrograms[i], (0, 0, 0, max_input_length - spectrograms[i].shape[0]), value=0)
        if len(labels[i]) < max_label_length:
            # pad labels to max_label_length
            labels[i] = torch.nn.functional.pad(
                labels[i], (0, max_label_length - len(labels[i])), value=0)

    # stack spectrograms and labels
    spectrograms = torch.stack(spectrograms, dim=0)
    spectrograms = spectrograms.unsqueeze(1).transpose(2, 3)
    labels = torch.stack(labels, dim=0)

    return spectrograms, labels, input_lengths, label_lengths


def greedyDecoder(output, blank_label=28):
    """
    decode a batch of utterances 
    arguments:
        output: network output tensor, shape B x T x C where B=batch_size, T=time_steps, C=characters
        blank_label: id of the blank label token
    returns:
        list of decoded strings
    """
    # Extract the most probable label for each time step
    _, predicted_labels = torch.max(output, dim=2)
    # merge repeated labels
    merged_labels = []
    for i in range(predicted_labels.shape[0]):
        merged_labels.append([])
        prev_label = -1
        for j in range(predicted_labels.shape[1]):
            if predicted_labels[i][j] != prev_label:
                merged_labels[i].append(predicted_labels[i][j])
                prev_label = predicted_labels[i][j]
    # convert to string and remove blank labels
    decoded_strings = []
    for i in range(len(merged_labels)):
        decoded_string = []
        for j in range(len(merged_labels[i])):
            if merged_labels[i][j] != blank_label:
                decoded_string.append(merged_labels[i][j])
        decoded_strings.append(intToStr(decoded_string))
    return decoded_strings


def levenshteinDistance(ref, hyp):
    """
    calculate levenshtein distance (edit distance) between two sequences
    arguments:
        ref: reference sequence
        hyp: sequence to compare against the reference
    output:
        edit distance (int)
    """
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,
                    d[i][j - 1] + 1,
                    d[i - 1][j - 1] + 1
                )
    return d[len(ref)][len(hyp)]
