import numpy as np

from lab2_tools import *


# already implemented
def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models
   
    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """
    num_states_hmm1 = len(hmm1['startprob'])-1
    num_states_hmm2 = len(hmm2['startprob'])-1
    num_states_concat = num_states_hmm1+num_states_hmm2+1
    
    startprob = np.concatenate((hmm1['startprob'],hmm2['startprob'][1:]))

    transmat = np.zeros((num_states_concat,num_states_concat))
    transmat[:num_states_hmm1+1,:num_states_hmm1+1] = hmm1['transmat']
    transmat[num_states_hmm1:,num_states_hmm1:] = hmm2['transmat']

    means = np.concatenate((hmm1['means'], hmm2['means']), axis=0)

    covars = np.concatenate((hmm1['covars'], hmm2['covars']), axis=0)

    concatenated_hmm = {'startprob': startprob,
                       'transmat': transmat,
                       'means': means,
                       'covars': covars}
    
    return concatenated_hmm


# already implemented, uses concatTwoHMMs()
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name. 
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1,len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat


def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """
    return np.log(np.sum(np.exp(log_emlik) * weights, axis=1))
    

def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    N, M = log_emlik.shape
    log_alpha = np.zeros((N, M))
    log_alpha[0] = log_startprob + log_emlik[0]
    for n in range(1, N):
        for j in range(M):
            log_alpha[n, j] = log_emlik[n, j] + logsumexp(log_alpha[n-1] + log_transmat[:, j])
    return log_alpha

    

def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """
    backward_prob = np.zeros(log_emlik.shape)
    N, M = log_emlik.shape
    backward_prob[-1] = 0
    for n in range(N-2, -1, -1):
        for i in range(M):
            backward_prob[n, i] = logsumexp(log_emlik[n+1] + log_transmat[i] + backward_prob[n+1])
    return backward_prob



def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    log_V = np.zeros(log_emlik.shape)
    N, M = log_emlik.shape
    
    log_V[0] = log_startprob + log_emlik[0]
    for n in range(1, N):
        for j in range(M):
            log_V[n, j] = log_emlik[n, j] + np.max(log_V[n-1] + log_transmat[:, j])    
    
    if forceFinalState:
        best_state = len(log_V[-1]) - 1
    else:
        best_state = np.argmax(log_V[-1])

    viterbi_loglik = log_V[-1][best_state]

    viterbi_path = np.zeros(N, dtype=int)
    viterbi_path[-1] = best_state
    
    for n in range(N-2, -1, -1):
        best_state = np.argmax(log_V[n])
        viterbi_path[n] = best_state
    
    return viterbi_loglik, viterbi_path


def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """
    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1)[:, np.newaxis]
    return log_gamma

def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """
    D = X.shape[1]
    M = log_gamma.shape[1]
    
    means = np.zeros((M, D))
    covars = np.zeros((M, D))
    # gamma = np.exp(log_gamma - logsumexp(log_gamma, axis=1))
    gamma = np.exp(log_gamma)
    
    for i in range(M):
        sum_gamma_i = np.sum(gamma[:, i])
        means[i] = np.sum(gamma[:, i][:, np.newaxis] * X, axis=0) / sum_gamma_i
        new_var = np.sum(gamma[:, i][:, np.newaxis] * (X - means[i]) ** 2, axis=0) / sum_gamma_i
        covars[i] = np.maximum(new_var, varianceFloor)
        
    return means, covars
    
    
