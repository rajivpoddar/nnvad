from mlp import MLP
from scipy.io import wavfile
import sys
import numpy as np
import theano
import theano.tensor as T
import string
import pysox
import random
import os
import argparse

# classifer has been trained on 8khz samples of 25ms length
SAMPLE_RATE = 8000 
WINDOW_SIZE = 0.025

def random_string():
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(6))

def downsample(fs, sig):
    in_file = random_string() + ".wav"
    out_file = random_string() + ".wav"

    wavfile.write(in_file, fs, sig) 

    sox_in = pysox.CSoxStream(in_file)
    sox_out = pysox.CSoxStream(out_file, 'w', pysox.CSignalInfo(SAMPLE_RATE, 1, 8), fileType='wav')
    sox_chain = pysox.CEffectsChain(sox_in, sox_out)
    sox_chain.add_effect(pysox.CEffect("rate", [str(SAMPLE_RATE)]))
    sox_chain.flow_effects()
    sox_out.close()

    fs, sig = wavfile.read(out_file)

    os.unlink(in_file)
    os.unlink(out_file)

    return sig

class MLP_VAD(object):
    def __init__(self, model_file):
        rng = np.random.RandomState(1234)

        self.x = T.matrix('x')

        self.classifier = MLP(
            rng=rng,
            input=self.x,
            n_in=200,
            n_hidden=180,
            n_out=2
        )

        self.classifier.load_model(model_file)

    def classify(self, fs, sig):
        if fs != SAMPLE_RATE:
            sig = downsample(fs, sig)

        sig = np.asarray(sig, dtype=np.float)/2**8

        num_samples = int(WINDOW_SIZE * SAMPLE_RATE)
        num_frames = len(sig)/num_samples
        sig = sig.reshape((num_frames, num_samples))
        spec = np.abs(np.fft.fft(sig)) # spectrum of signal

        shared_x = theano.shared(np.asarray(spec, dtype=theano.config.floatX), borrow=True)

        index = T.lscalar()  # index to a [mini]batch

        predict_model = theano.function(
            inputs=[index],
            outputs=self.classifier.y_pred,
            givens={
                self.x: shared_x[index:index + 1],
            }
        )

        # classify each frame
        predicted_values = [predict_model(i)[0] for i in xrange(num_frames)]

        # classifier returns 0 (noise) or 1 (speech) for each frame
        # the mean of all frames is our final result
        speech_prob = np.round(np.mean(predicted_values), 2)

        return speech_prob

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Voice Activity Detection using Theano')
    parser.add_argument('input_file', action='store', type=str, help='segment of input file of 200ms duration')
    parser.add_argument('-m, --model-file', action='store', type=str, dest='model_file', help='the model file to use', default='models/params.pkl')
    parser.add_argument('-t, --noise-threshold', action='store', type=float, dest='noise_threshold', help='noise thresold (default: 0.25)', default=0.25)
    args = parser.parse_args()

    fs, sig = wavfile.read(args.input_file)

    mlp = MLP_VAD(args.model_file)
    speech_prob = mlp.classify(fs, sig)

    if speech_prob < args.noise_threshold:
        print "noise (%.2f)" % (speech_prob)
    else:
        print "speech (%.2f)" % (speech_prob)
