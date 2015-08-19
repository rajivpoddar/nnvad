#!/usr/bin/python

import cPickle
from os import listdir, path
import numpy as np
from scipy.io import wavfile
import sys
import gzip
import random

def build_set(files):
    size = len(files)
    inputs = np.zeros((size, 200))
    targets = np.zeros(size, dtype=np.int)

    for i in range(size):
        fn = files[i]
        fs, sig = wavfile.read('data/audio/' + fn)
        sig = np.asarray(sig, dtype=np.float)/2**8
        spec = np.abs(np.fft.fft(sig))
        inputs[i] = spec
        if fn[0] == 's':
            targets[i] = 1
        else:
            targets[i] = 0

    return (inputs, targets)

def main():
    files = listdir('data/audio/')

    mtimes = []
    for f in files:
        mtime = path.getmtime('data/audio/' + f)
        mtimes.append([f, mtime])

    files = sorted(mtimes, key=lambda m: m[1], reverse=False)
    files = [f[0] for f in files]

    n_files = [f for f in files if f[0] == 'n']
    s_files =  [f for f in files if f[0] == 's']

    audio_files = []
    n_index = 0
    s_index = 0
    for i in range(20000):
        if i%2 == 0:
            audio_files.append(n_files[n_index])
            n_index = n_index + 1
        else:
            audio_files.append(s_files[s_index])
            s_index = s_index + 1

    test_set = build_set(audio_files[0:3000])
    valid_set = build_set(audio_files[3000:6000])
    train_set = build_set(audio_files[6000:len(audio_files)])

    dataset = (train_set, valid_set, test_set)

    with gzip.open('data/audio.pkl.gz', 'wb') as f:
        cPickle.dump(dataset, f, cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()

