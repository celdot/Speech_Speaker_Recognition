# DT2119, Lab 4 End-to-end Speech Recognition

import argparse
import itertools
import os

import kenlm
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchaudio
from pyctcdecode import build_ctcdecoder
from torch import nn
from tqdm import tqdm

from lab4_proto import *

torch.backends.cudnn.enabled = False

"""
HYPERPARAMETERS
"""
hparams = {
    "n_cnn_layers": 3,
    "n_rnn_layers": 5,
    "rnn_dim": 512,
    "n_class": 29,
    "n_feats": 80,
    "stride": 2,
    "dropout": 0.1,
    "learning_rate": 5e-4,
    "batch_size": 16,
    "epochs": 3
}


"""
MODEL DEFINITION
"""


class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""

    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        # (batch, channel, feature, time)
        return x.transpose(2, 3).contiguous()


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
            except with layer norm instead of batch norm
    """

    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels,
                              kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels,
                              kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x  # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # print('bi-gru, in:',x.shape)
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        # print('bi-gru, out:',x.shape)
        return x


class SpeechRecognitionModel(nn.Module):
    """Speech Recognition Model Inspired by DeepSpeech 2"""

    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//stride
        # cnn for extracting heirachal features
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1,
                        dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i == 0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i == 0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class),
            nn.LogSoftmax(dim=2)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2],
                   sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x


"""
ACCURACY MEASURES
"""


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)
    edit_distance = levenshteinDistance(ref_words, hyp_words)
    ref_len = len(ref_words)

    if ref_len > 0:
        wer = float(edit_distance) / ref_len
    else:
        raise ValueError("empty reference string")
    return wer


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = levenshteinDistance(reference, hypothesis)
    ref_len = len(reference)
    if ref_len > 0:
        cer = float(edit_distance) / ref_len
    else:
        raise ValueError("empty reference string")
    return cer


"""
TRAINING AND TESTING
"""


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    data_len = len(train_loader.dataset)
    for batch_idx, _data in tqdm(enumerate(train_loader), desc="Training", total=len(train_loader)):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()
        # model output is (batch, time, n_class)
        output = model(spectrograms)
        # transpose to (time, batch, n_class) in loss function
        loss = criterion(output.transpose(0, 1), labels,
                         input_lengths, label_lengths)
        loss.backward()
        optimizer.step()
        if epoch == 0 and (batch_idx % 100 == 0 or batch_idx == data_len):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(spectrograms), data_len,
                100. * batch_idx / len(train_loader), loss.item()))
            
    return loss.item()


def test(model, device, test_loader, criterion, epoch):
    print('\nevaluating…')
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with torch.no_grad():
        for I, _data in tqdm(enumerate(test_loader), desc="Testing", total=len(test_loader)):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            # model output is (batch, time, n_class)
            output = model(spectrograms)
            # transpose to (time, batch, n_class) in loss function
            loss = criterion(output.transpose(0, 1), labels,
                             input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)

            # get target text
            decoded_targets = []
            for i in range(len(labels)):
                decoded_targets.append(
                    intToStr(labels[i][:label_lengths[i]].tolist()))

            # get predicted text
            decoded_preds = greedyDecoder(output)

            # calculate accuracy
            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)
    return avg_cer, avg_wer, test_loss

"""
GRID SEARCH FOR LANGUAGE MODEL PARAMETERS
"""

def grid_search_lm_params(model, device, val_loader, root_dir):
    print("Starting grid search over alpha and beta...")

    alphas = [0.2, 0.5, 1.0, 1.5]
    betas = [0.5, 1.0, 1.5, 2.0]
    kenlm_model_path = os.path.join(root_dir, "wiki-interpolate.3gram.arpa")

    best_wer = float("inf")
    best_alpha = None
    best_beta = None

    model.eval()
    with torch.no_grad():
        for alpha, beta in itertools.product(alphas, betas):
            decoder = build_ctcdecoder(
                [chr(i + 97) for i in range(26)] + ["'", "_"],
                kenlm_model_path=kenlm_model_path,
                alpha=alpha,
                beta=beta
            )

            total_wer = []
            for data in val_loader:
                spectrograms, labels, _, label_lengths = data
                spectrograms, labels = spectrograms.to(device), labels.to(device)
                output = model(spectrograms)

                decoded_targets = [intToStr(labels[i][:label_lengths[i]].tolist())
                                   for i in range(len(labels))]

                for i in range(len(spectrograms)):
                    pred = decoder.decode(output[i].cpu().detach().numpy())
                    total_wer.append(wer(decoded_targets[i], pred))

            avg_wer = sum(total_wer) / len(total_wer)
            print(f"alpha={alpha}, beta={beta}, WER={avg_wer:.4f}")

            if avg_wer < best_wer:
                best_wer = avg_wer
                best_alpha, best_beta = alpha, beta

    print(f"\nBest WER: {best_wer:.4f} with alpha={best_alpha} and beta={best_beta}")



"""
MAIN PROGRAM
"""
def main(root_dir, mode, model_load, wavfiles, use_language_model=False, grid_search=False):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device:', device)

    print("don't download train")
    #train_dataset = torchaudio.datasets.LIBRISPEECH(
    #    root_dir, url='train-clean-100', download=True)
    val_dataset = torchaudio.datasets.LIBRISPEECH(
        root_dir, url='dev-clean', download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH(
        root_dir, url='test-clean', download=True)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # train_loader = data.DataLoader(dataset=train_dataset,
    #                                 batch_size=hparams['batch_size'],
    #                                 shuffle=True,
    #                                 collate_fn=lambda x: dataProcessing(
    #                                    x, train_audio_transform),
    #                                 **kwargs)

    val_loader = data.DataLoader(dataset=val_dataset,
                                    batch_size=hparams['batch_size'],
                                    shuffle=True,
                                    collate_fn=lambda x: dataProcessing(
                                        x, test_audio_transform),
                                    **kwargs)

    test_loader = data.DataLoader(dataset=test_dataset,
                                    batch_size=hparams['batch_size'],
                                    shuffle=False,
                                    collate_fn=lambda x: dataProcessing(
                                        x, test_audio_transform),
                                    **kwargs)

    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'],
        hparams['n_rnn_layers'],
        hparams['rnn_dim'],
        hparams['n_class'],
        hparams['n_feats'],
        hparams['stride'],
        hparams['dropout']
    ).to(device)

    # print(model)
    # print('Num Model Parameters', sum(
    #     [param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=28).to(device)

    print(mode)

    if model_load != '':
        print(model_load)
        model.load_state_dict((torch.load(model_load, weights_only=True)))

    if mode == 'train':
        best_cer = float("inf")
        for epoch in range(hparams['epochs']):
            train_loss = train(model, device, train_loader, criterion, optimizer, epoch)
            print('Epoch:', epoch, 'Train Loss:', train_loss)
            avg_cer, avg_wer, _= test(model, device, val_loader, criterion, epoch)
            print('Epoch:', epoch, 'Validation CER:', avg_cer, 'Validation WER:', avg_wer)
            if avg_cer < best_cer:
                best_cer = avg_cer
                torch.save(model.state_dict(), os.path.join(root_dir, 'best_model.pth'))

    elif mode == 'test':
        avg_cer, avg_wer, test_loss = test(model, device, test_loader, criterion, -1)
        print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(
        test_loss, avg_cer, avg_wer))
        
    if use_language_model:
        # Path to your ARPA language model
        kenlm_model_path = os.path.join(root_dir, "wiki-interpolate.3gram.arpa")
        decoder_labels = [
                "'", "_", " ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
                "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
            ]

        decoder = build_ctcdecoder(
            decoder_labels,
            kenlm_model_path=kenlm_model_path,
            alpha=0.5,  # LM weight
            beta=1.0    # Word insertion bonus
        )
        
    if grid_search:
        grid_search_lm_params(model, device, val_loader, root_dir)

    elif mode == 'recognize':
        for wavfile in wavfiles:
            waveform, _ = torchaudio.load(wavfile, normalize=True)
            spectrogram = test_audio_transform(waveform)
            wav_input = torch.unsqueeze(spectrogram, dim=0).to(device)
            output = model(wav_input)
            if use_language_model:
                text = decoder.decode(output[0].cpu().detach().numpy())
            else:
                text = greedyDecoder(output)
            print('wavfile:', wavfile)
            print('text:', text)
            
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mode', help='train, test or recognize')
    argparser.add_argument('--model', type=str,
                           help='model to load', default='')
    argparser.add_argument('wavfiles', nargs='*', help='wavfiles to recognize')
    argparser.add_argument('--use_language_model', action='store_true',
                        help='use language model for decoding')
    argparser.add_argument('--grid_search', action='store_true',
                        help='perform grid search on alpha/beta for language model')

    args = argparser.parse_args()
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # print cuda device 
    if torch.cuda.is_available():
        print('Using GPU:', torch.cuda.get_device_name(0))
    else:
        print('Using CPU')
    print('ROOT_DIR:', ROOT_DIR)
    
    main(ROOT_DIR, args.mode, args.model, args.wavfiles, args.use_language_model, args.grid_search)


