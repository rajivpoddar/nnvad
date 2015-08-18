from __future__ import division
import sys
from scipy.io import wavfile
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Find lowest energy points')
parser.add_argument('input_file', action='store', type=str, help='input file to process')
parser.add_argument('-n, --num-results', action='store', type=int, dest='num_results', help='number of results (default: 10)', default=10)
parser.add_argument('-s, --step-size', action='store', type=int, dest='step_size', help='step size (default: 1)', default=1)
parser.add_argument('-w, --window-size', action='store', type=float, dest='window_size', help='hamming window size (default: 0.01ms)', default=0.01)
args = parser.parse_args()

Fs, signal = wavfile.read(args.input_file)
signal = signal / max(abs(signal)) 

sampsPerMilli = Fs / 1000
millisPerFrame = int(args.window_size * 1000)
sampsPerFrame = int(sampsPerMilli * millisPerFrame)
nFrames = int(len(signal) / sampsPerFrame)

STEs = [] 
for k in range(nFrames):
    startIdx = k * sampsPerFrame
    stopIdx = startIdx + sampsPerFrame
    window = signal[startIdx:stopIdx]
    STE = np.sum(window ** 2) / np.float64(len(window))
    STEs.append(STE)

F = np.sort(STEs)[0:args.num_results*args.step_size:args.step_size]
seconds = [[np.where(STEs == e)[0][0] * millisPerFrame/1000, e] for e in F]

seconds = np.array(seconds)
for s in seconds[seconds[:,1].argsort()]:
    print "%.2f %.7f" % (s[0], s[1])
