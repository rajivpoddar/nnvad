#!/usr/bin/python

from __future__ import division
import numpy as np
import math
from scipy.special import jv, expn
from scikits.audiolab import Sndfile, Format
import argparse
import sys
from mlp_vad import MLP_VAD

np.seterr('ignore')

def MMSESTSA(signal, fs, W, mlp, saved_params=None):
    SP = 0.5
    wnd = np.hamming(W)

    y = segment(signal, W, SP, wnd)
    Y = np.fft.fft(y, axis=0)
    YPhase = np.angle(Y[0:len(Y)/2+1,:])
    Y = np.abs(Y[0:len(Y)/2+1,:])
    numberOfFrames = Y.shape[1]

    NoiseLength = 1
    alpha = 0.75

    if saved_params == None:
        N = np.ones(Y[:,0].shape)
        LambdaD = np.ones(Y[:,0].shape)
    else:
        N = saved_params['N']
        LambdaD = saved_params['LambdaD']

    G = np.ones(N.shape)
    Gamma = G

    Gamma1p5 = math.gamma(1.5)
    X = np.zeros(Y.shape)

    sig = y.T.flatten()
    vad = mlp.classify(fs, sig)
    vad = vad[0:Y.shape[1]*2]
    vad = vad.reshape((len(vad)/2, 2))

    for i in range(numberOfFrames):
        Y_i = Y[:,i]

        if vad[i].all() == 0:
            N = (NoiseLength * N + Y_i) / (NoiseLength + 1)
            LambdaD = (NoiseLength * LambdaD + (Y_i ** 2)) / (1 + NoiseLength)

        gammaNew = (Y_i ** 2) / LambdaD
        xi = alpha * (G ** 2) * Gamma + (1 - alpha) * np.maximum(gammaNew - 1, 0)

        Gamma = gammaNew
        nu = Gamma * xi / (1 + xi)

        G = (Gamma1p5 * np.sqrt(nu)) / Gamma * np.exp(-1 * nu / 2) * ((1 + nu) * bessel(0, nu / 2) + nu * bessel(1, nu / 2))
        Indx = np.isnan(G) | np.isinf(G)
        G[Indx] = xi[Indx] / (1 + xi[Indx])

        X[:,i] = G * Y_i

    output = OverlapAdd2(X, YPhase, W, SP * W)
    return output, {'N': N, 'LambdaD': LambdaD}

def OverlapAdd2(XNEW, yphase, windowLen, ShiftLen):
    FrameNum = XNEW.shape[1]
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
    N = int(np.fix(L-W)/SP) + 1

    Window = Window.flatten(1)

    Index = (np.tile(np.arange(0,W), (N,1)) + np.tile(np.arange(0,N) * SP, (W,1)).T).T
    hw = np.tile(Window, (N, 1)).T
    Seg = signal[Index] * hw
    return Seg

def bessel(v, X):
    return ((1j**(-v))*jv(v,1j*X)).real

# main

parser = argparse.ArgumentParser(description='Speech enhancement/noise reduction using MMSE STSA algorithm and an MLP VAD')
parser.add_argument('input_file', action='store', type=str, help='input file to clean')
parser.add_argument('output_file', action='store', type=str, help='output file to write (default: stdout)', default=sys.stdout)
parser.add_argument('-m, --model-file', action='store', type=str, dest='model_file', help='model file to use (default: models/params.pkl)', default='models/params.pkl')
args = parser.parse_args()

input_file = Sndfile(args.input_file, 'r')

fs = input_file.samplerate
num_frames = input_file.nframes

window_size = int(0.05*fs) # 50ms

mlp = MLP_VAD(args.model_file)

output_file = Sndfile(args.output_file, 'w', Format(type=input_file.file_format, encoding='pcm16', endianness=input_file.endianness), input_file.channels, fs)

chunk_size = int(np.round(fs*60))
saved_params = None

frames_read = 0
while (frames_read < num_frames):
    if frames_read + chunk_size > num_frames:
        chunk_size = num_frames - frames_read

    signal = input_file.read_frames(chunk_size)
    frames_read = frames_read + chunk_size

    output, saved_params = MMSESTSA(signal, fs, window_size, mlp, saved_params)

    output = np.array(output*np.iinfo(np.int16).max, dtype=np.int16)
    output_file.write_frames(output)

input_file.close()
output_file.close()
