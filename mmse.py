# References:
# [1] MMSE STSA by Esfandiar Zavarehei Available: http://www.mathworks.com/matlabcentral/fileexchange/10143-mmse-stsa
# [2] Speech Enhancement Using a- Minimum Mean-Square Error Short-Time Spectral Amplitude Estimator Eprahim Y. and Malah D.

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.special import *
from scipy.io import wavfile
import matplotlib.mlab as mlab
import sys
import scikits.audiolab

np.seterr('ignore')

def segment_windows(signal, ww, ov):
    '''
    Parameters
        signal  - array of normalized signal samples
        ww - window width
        ov - overlap ratio of windows
    '''
    l = len(signal)
    d = 1 - ov
    frames = int(np.floor((l - ww) / ww / d) + 1)
    seg = np.zeros((ww, frames))

    for i in range(frames):
        start = i * ww * ov
        stop = start + ww
        s = signal[start:stop] * np.hamming(ww)
        seg[:, i] = s

    return seg, frames

def combine_segments(segments, ov):
    '''
    Parameters
        signal  - array of normalized signal samples
        ov - overlap ratio of windows
    '''
    ww, frames = segments.shape
    dataleng = ww * (1 - ov) * (frames - 1) + ww

    sig = np.zeros(dataleng)
    for i in range(frames):
        start = i * ww * (1 - ov)
        stop = start + ww
        sig[start:stop] = sig[start:stop] + segments[:, i]
    return sig

def VAD(frame,noise,nc=0,nm=3,h=8):
    '''
    Parameters
        frame - current signal frame
        noise - noise estimate
        nc - number of previous noise frames
        nm - noise spectral threshold
        h - number of noise frames to reset
    '''

    specdist = 20*(np.log10(frame)-np.log10(noise))
    specdist[np.where(specdist<0)]=0

    dist = np.mean(specdist,axis = 0)

    if(dist< nm):
        noise_f = 1
        nc +=1
    else:
        noise_f = 0
        nc = 0

    if(nc>h):
        speech_f = 0
    else:
        speech_f=1

    return noise_f, speech_f, nc, dist

def bessel(v,X):
    return ((1j**(-v))*jv(v,1j*X)).real

#flags
speech_f = 0
noise_f  = 0
show_graphs = False

#cfs, clean = wavfile.read('./data/car_clean_lom.wav')
fs, ss = wavfile.read(sys.argv[1])  # sampled signal and frequency
ss = ss / np.power(2, 15)


# User Parameters
# FFT
window_length = 1024 
overlap_ratio = 0.5  # [0:0.5]

#inital no speech duration + segments
nsd = 0.1
nss= np.floor((nsd*fs - window_length)/(overlap_ratio*window_length)+1)

#smoothing
alpha = 0.99

#Noise characteristics
nc = 0 # counter
nl = 9 # length
nt = 10 # noise threshold


ssw, frames = segment_windows(ss, window_length, overlap_ratio)
dataleng = window_length * (1 - overlap_ratio) * (frames - 1) + window_length

sfft = np.fft.fft(ssw, axis=0)
    
sfftmag = np.abs(sfft)
sfftphase = np.zeros(sfft.shape)
for i in range(frames):
    sfftphase[:, i] = np.angle(sfft[:, i])

fr = sfftmag.shape[0]

nm = np.mean(sfftmag[:,:nss],axis = 1)
lmda_D = np.mean(np.power(sfftmag[:,:nss],2),axis = 1)

'''
# noiseprofile
n_fs, n_ss = wavfile.read('noise2.wav')  # sampled signal and frequency
n_ss = n_ss / np.power(2, 15)
n_ssw, n_frames = segment_windows(n_ss, window_length, 0)
n_sfft = np.fft.fft(n_ssw, axis=0)
n_sfftmag = np.abs(n_sfft)
nm = np.mean(n_sfftmag[:,:nss],axis = 1)
lmda_D = np.mean(np.power(n_sfftmag[:,:nss],2),axis = 1)
# noiseprofile
'''


k = np.ones(nm.shape)
gamma = k

g1_5 = math.gamma(1.5)

es = np.zeros(sfftmag.shape)

for i in range(frames):
    if(i<nss):
        speech_f = 0
        nc = 100
    else:
        noise_f, speech_f, nc, dist = VAD(sfftmag[:,i],nm,nc,nt)

    if(speech_f == 0):
        nm = (nl*nm+sfftmag[:,i])/(nl+1)
        lmda_D = (nl*lmda_D + np.power(sfftmag[:,i],2))/(nl+1)

    gammaNew = np.power(sfftmag[:,i],2)/lmda_D
    xi = alpha*np.power(k,2)*gamma+(1-alpha)*np.maximum(gammaNew-1,0)
    gamma = gammaNew
    nu = gamma*xi/(1+xi)
    k = (g1_5*np.sqrt(nu))/gamma*np.exp(-1*nu/2)*((1+nu)*bessel(0,nu/2)+nu*bessel(1,nu/2))
    #k = (xi/(1+xi))*np.exp(0.5*expn(1, nu))

    inf = np.isnan(k)
    k[inf] = xi[inf]/(1+xi[inf])
    es[:,i] = k*sfftmag[:,i]


estim_spec = es * np.exp(1j * sfftphase)
estim_seg = np.real(np.fft.ifft(estim_spec, axis=0))
estim = combine_segments(estim_seg, overlap_ratio)

#scikits.audiolab.play(estim, fs)
estim = np.array(estim * np.iinfo(np.int16).max, dtype=np.int16);
wavfile.write(sys.argv[2], fs, estim)

if(show_graphs):
    t = np.linspace(0, dataleng, dataleng) / fs
    plt.subplot(3, 3, 1),

    plt.plot(t, clean[:dataleng])
    plt.xlabel('Time (s)')
    plt.title('Clean speech')
    plt.xlim(0, max(t))
    plt.subplot(3, 3, 4)
    plt.plot(t, ss[:dataleng])
    plt.xlabel('Time (s)')
    plt.title('Observed speech')
    plt.xlim(0, max(t))
    plt.subplot(3, 3, 7),
    plt.plot(t, estim)
    plt.xlabel('Time (s)')
    plt.title('Denoised speech')
    plt.xlim(0, max(t))

    cfft = np.abs(np.fft.fft(clean))
    ssfft = np.abs(np.fft.fft(ss))
    estimfft = np.abs(np.fft.fft(estim))
    plt.subplots_adjust(hspace=0.5)

    plt.subplot(3, 3, 2)
    plt.plot(np.arange(len(cfft)) / len(cfft), cfft)
    plt.xlabel('$f_w$')
    plt.title('Clean speech')
    plt.subplot(3, 3, 3)
    Pxx, freqs, bins, im = plt.specgram(clean, Fs=fs)
    plt.xlim(0, max(bins))
    plt.ylabel("Frequency (Hz)")
    plt.xlabel('t (s)')
    plt.title('Clean speech')
    plt.subplot(3, 3, 5)
    plt.xlabel('$f_w$')
    plt.title('Observed speech')
    plt.plot(np.arange(len(ssfft)) / len(ssfft), ssfft)

    plt.subplot(3, 3, 6)
    Pxx, freqs, bins, im = plt.specgram(ss, Fs=fs)
    plt.xlim(0, max(bins))
    plt.ylabel("Frequency (Hz)")
    plt.xlabel('t (s)')
    plt.title('Observed speech')
    plt.subplot(3, 3, 8)
    plt.xlabel('$f_w$')
    plt.title('Denoised speech')
    plt.plot(np.arange(len(estimfft)) / len(estimfft), estimfft)
    plt.subplot(3, 3, 9)
    Pxx, freqs, bins, im = plt.specgram(estim, Fs=fs)
    plt.xlim(0, max(bins))
    plt.ylabel("Frequency (Hz)")
    plt.xlabel('t (s)')
    plt.title('Denoised speech')
    plt.show()
