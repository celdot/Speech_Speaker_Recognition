import numpy as np
from lab2_tools import log_multivariate_normal_density_diag
from lab2_proto import viterbi
from lab3_tools import *


def words2phones(wordList, pronDict, addSilence=True, addShortPause=True):
    """ word2phones: converts word level to phone level transcription adding silence

    Args:
        wordList: list of word symbols
        pronDict: pronunciation dictionary. The keys correspond to words in wordList
        addSilence: if True, add initial and final silence
        addShortPause: if True, add short pause model "sp" at end of each word
    Output:
        list of phone symbols
        """
    phoneList = []
    for i, word in enumerate(wordList):
        if word in pronDict:
            phoneList += pronDict[word]
            if addShortPause and i < len(wordList) - 1:
                phoneList.append('sp')
        else:
            print("Warning: word not found in pronunciation dictionary: ", word)
    if addSilence:
        phoneList = ["sil"] + phoneList + ["sil"]
    return phoneList


def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
    """ forcedAlignmen: aligns a phonetic transcription at the state level

    Args:
        lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
                 computed the same way as for the training of phoneHMMs
        phoneHMMs: set of phonetic Gaussian HMM models
        phoneTrans: list of phonetic symbols to be aligned including initial and
                        final silence

    Returns:
        list of strings in the form phoneme_index specifying, for each time step
        the state from phoneHMMs corresponding to the viterbi path.
    """
    obsloglik = log_multivariate_normal_density_diag(lmfcc, phoneHMMs['means'], phoneHMMs['covars'])
    _, path, _ = viterbi(
        obsloglik,
        np.log(phoneHMMs['startprob'][:-1]),
        np.log(phoneHMMs['transmat'][:-1, :-1])
    )
    return [phoneTrans[i] for i in path]
    


