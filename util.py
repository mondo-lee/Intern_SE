import librosa
import numpy as np
# from pypesq import pesq
from pesq import pesq
from pystoi.stoi import stoi

'''
respective pesq and stoi related libraries.
pesq is a more regularly updated version compared to pypesq
'''

import scipy
import pdb
import torch, os
# import torch.nn.functional as F
# from torchaudio import functional as taF
### figuring these out LATER

epsilon = np.finfo(float).eps
### difference between 1.0 and next smallest representation larger than 1.0



def check_path(path):
    if not os.path.isdir(path): 
        os.makedirs(path) ### creates leaf directory and all intermediate ones
        
def check_folder(path):
    path_n = '/'.join(path.split('/')[:-1])
    check_path(path_n)

def cal_score(clean,enhanced):
    clean = clean/abs(clean).max()
    enhanced = enhanced/abs(enhanced).max()
    s_stoi = stoi(clean, enhanced, 16000)
    # s_pesq = pesq(clean, enhanced, 16000)
    s_pesq = pesq(16000,clean, enhanced, 'wb')
    
    return round(s_pesq,5), round(s_stoi,5)


def get_filepaths(directory,ftype='.wav'): ### allows use of other file formats
    file_paths = []
    for root, directories, files in os.walk(directory): ### ??? tree traversal?
        for filename in files:
            if filename.endswith(ftype):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.

    return sorted(file_paths)

def make_spectrum(filename=None, y=None, is_slice=False, feature_type='log1p', mode=None, FRAMELENGTH=None,
                 SHIFT=None, _max=None, _min=None): ### is_slice, FRAMELENGTH, SHIFT do not appear to be used
    '''
    Uses capabilities of librosa to load, STFT, and scale the spectrum.
    '''
    if y is not None:
        y = y
    else:
        y, sr = librosa.load(filename, sr=16000)
        if sr != 16000:
            raise ValueError('Sampling rate is expected to be 16kHz!')
        if y.dtype == 'int16':
            y = np.float32(y/32767.)
        elif y.dtype !='float32':
            y = np.float32(y)

    # Normalize waveform
    # y = y / np.max(abs(y)) / 2.

    D = librosa.stft(y,center=False, n_fft=512, hop_length=256,win_length=512,window=scipy.signal.hamming)
    utt_len = D.shape[-1]
    phase = np.exp(1j * np.angle(D))
    D = np.abs(D)

    # Feature type
    if feature_type == 'log1p':
        Sxx = np.log1p(D)
    elif feature_type == 'lps': ### log power spectrum
        Sxx = np.log10(D**2)
    elif feature_type == 'lps+':
        Sxx = np.log10((D+1e-12)**2)
    else:
        Sxx = D

    if mode == 'mean_std': ### normalize
        mean = np.mean(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))
        std = np.std(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))+1e-12
        Sxx = (Sxx-mean)/std  
    elif mode == 'minmax':
        Sxx = 2 * (Sxx - _min)/(_max - _min) - 1 ### how to obtain _min and _max if not from Librosa's STFT?

    return Sxx, phase, len(y)

def recons_spec_phase(Sxx_r, phase, length_wav, feature_type='log1p'):
    if feature_type == 'log1p':
        Sxx_r = np.expm1(Sxx_r) ### inverse of log1p
        if np.min(Sxx_r) < 0:
            print("Expm1 < 0 !!")
        # Sxx_r = np.clip(Sxx_r, a_min=0., a_max=None)
    elif feature_type == 'lps':
        Sxx_r = np.sqrt(10**(Sxx_r)) ### inverse of lps

    R = np.multiply(Sxx_r , phase)
    
#     pdb.set_trace()
    result = librosa.istft(R,
                     center=False,
                     hop_length=256,
                     win_length=512,
                     window=scipy.signal.hamming,
                     length=length_wav)
    return result
