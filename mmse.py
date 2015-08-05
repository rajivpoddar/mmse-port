from __future__ import division
import numpy as np
import math
from scipy.special import *
from scipy.io import wavfile
from numpy.matlib import repmat
from scipy.signal import lfilter
import sys

np.seterr('ignore')

def MMSESTSA85(signal, fs, IS=0.25):
    #W = int(np.fix(0.025 * fs))
    W = 1024 * 1

    SP = 0.4
    wnd = np.hamming(W)

    NIS = int(np.fix(((IS * fs - W) / (SP * W) + 1)))

    y = segment(signal, W, SP, wnd)
    Y = np.fft.fft(y, axis=0)
    YPhase = np.angle(Y[0:int(np.fix(len(Y)/2))+1,:])
    Y = np.abs(Y[0:int(np.fix(len(Y)/2))+1,:])
    numberOfFrames = Y.shape[1]

    N = np.mean(Y[:,0:NIS].T).T
    LambdaD = np.mean((Y[:,0:NIS].T) ** 2).T
    alpha = 0.99

    NoiseCounter = 0
    NoiseLength = 9

    G = np.ones(N.shape)
    Gamma = G

    Gamma1p5 = math.gamma(1.5)
    X = np.zeros(Y.shape)

    for i in range(numberOfFrames):
        if i < NIS:
            SpeechFlag = 0
            NoiseCounter = 100
        else:
            NoiseFlag, SpeechFlag, NoiseCounter, Dist = vad(Y[:,i], N, NoiseCounter)

        if SpeechFlag == 0:
            N = (NoiseLength * N + Y[:,i]) / (NoiseLength + 1)
            LambdaD = (NoiseLength * LambdaD + (Y[:,i] ** 2)) / (1 + NoiseLength)

        gammaNew = (Y[:,i] ** 2) / LambdaD
        xi = alpha * (G ** 2) * Gamma + (1 - alpha) * np.maximum(gammaNew - 1, 0)

        Gamma = gammaNew
        nu = Gamma * xi / (1 + xi)

        G = (xi/(1 + xi)) * np.exp(0.5 * expn(1, nu))

        #G = (Gamma1p5 * np.sqrt(nu)) / Gamma * np.exp(-1 * nu / 2) * ((1 + nu) * bessel(0, nu / 2) + nu * bessel(1, nu / 2))
        #Indx = np.isnan(G) | np.isinf(G)
        #G[Indx] = xi[Indx] / (1 + xi[Indx])

        X[:,i] = G * Y[:,i]

    output = OverlapAdd2(X, YPhase, W, SP * W)
    return output

def OverlapAdd2(XNEW, yphase, windowLen, ShiftLen):
    FreqRes, FrameNum = XNEW.shape
    Spec = XNEW * np.exp(1j * yphase)

    ShiftLen = int(np.fix(ShiftLen))

    if windowLen % 2:
        Spec = np.concatenate((Spec, np.flipud(np.conj(Spec[1:,]))))
    else:
        Spec = np.concatenate((Spec, np.flipud(np.conj(Spec[1:-1,:]))))

    sig = np.zeros(((FrameNum - 1) * ShiftLen + windowLen, 1)) 

    for i in range(FrameNum):
        start = i * ShiftLen
        spec = Spec[:,[i]]
        sig[start:start + windowLen] = sig[start:start + windowLen] + np.real(np.fft.ifft(spec, axis=0))

    return sig

def segment(signal, W, SP, Window):
    L = len(signal)
    SP = int(np.fix(W * SP))
    N = int(np.fix((L-W)/SP + 1))

    Window = Window.flatten(1)

    Index = (np.tile(np.arange(1,W+1), (N,1)) + np.tile(np.arange(0,N) * SP, (W,1)).T).T
    hw = np.tile(Window, (N, 1)).T
    Seg = signal[Index] * hw
    return Seg

def vad(signal, noise, NoiseCounter, NoiseMargin = 3, Hangover = 8):
    SpectralDist = 20 * (np.log10(signal) - np.log10(noise))
    SpectralDist[SpectralDist < 0] = 0

    Dist = np.mean(SpectralDist)
    if (Dist < NoiseMargin):
        NoiseFlag = 1
        NoiseCounter = NoiseCounter + 1
    else:
        NoiseFlag = 0
        NoiseCounter = 0

    if (NoiseCounter > Hangover):
        SpeechFlag=0
    else:
        SpeechFlag=1

    return NoiseFlag, SpeechFlag, NoiseCounter, Dist

def bessel(v, X):
    return ((1j**(-v))*jv(v,1j*X)).real

# main

fs, signal = wavfile.read(sys.argv[1])
dt = signal.dtype
signal = np.array(signal/(np.iinfo(dt).max), dtype='float')

output = MMSESTSA85(signal, fs)

output = np.array(output*np.iinfo(dt).max, dtype=dt)
wavfile.write(sys.argv[2], fs, output)
