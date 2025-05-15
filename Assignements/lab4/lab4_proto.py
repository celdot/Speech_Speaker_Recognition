import torch
from torchaudio.transforms import MelSpectrogram, FrequencyMasking, TimeMasking, Resample
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
    res = []
    for i in range(len(labels)):
        if labels[i] == 0:
            res.append("'")
        elif labels[i] == 1:
            res.append("_")
        else:
            res.append(chr(labels[i] - 2 + ord('a')))
    return ' '.join(res)

def strToInt(text):
    """
        convert string to list of integers
    Args:
        text: string
    Returns:
        list of ints
    """
    text = text.replace(" ", "_").lower()
    res = []
    for i in range(len(text)):
        if text[i] == "'":
            res.append(0)
        elif text[i] == "_":
            res.append(1)
        else:
            res.append(ord(text[i]) - ord('a') + 2)
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

    print(len(spectrograms))
    print(len(labels))
        
    max_input_length = max(input_lengths)
    max_label_length = max(label_lengths)
    for i in range(len(spectrograms)):
        if spectrograms[i].shape[0] < max_input_length:
            # pad spectrograms to max_input_length
            spectrograms[i] = torch.nn.functional.pad(
                spectrograms[i], (0, 0, 0, max_input_length - spectrograms[i].shape[0]), value=0)
            print(spectrograms[i].shape)
        if len(labels[i]) < max_label_length:
            # pad labels to max_label_length
            labels[i] = torch.nn.functional.pad(
                labels[i], (0, max_label_length - len(labels[i])), value=0)
            print(labels[i].shape)
    
    # stack spectrograms and labels
    spectrograms = torch.stack(spectrograms, dim=0)
    spectrograms = spectrograms.unsqueeze(1).transpose(2, 3)
    labels = torch.stack(labels, dim=0)
    
    
    
    
    
    
def greedyDecoder(output, blank_label=28):
    """
    decode a batch of utterances 
    arguments:
        output: network output tensor, shape B x T x C where B=batch_size, T=time_steps, C=characters
        blank_label: id of the blank label token
    returns:
        list of decoded strings
    """

def levenshteinDistance(ref,hyp):
    """
    calculate levenshtein distance (edit distance) between two sequences
    arguments:
        ref: reference sequence
        hyp: sequence to compare against the reference
    output:
        edit distance (int)
    """
