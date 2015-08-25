import numpy as np
import math
from scikits.audiolab import Sndfile, Format
import argparse
import sys
from subprocess import Popen, PIPE
import pysox
import os
from mlp_vad import MLP_VAD, random_string, SAMPLE_RATE

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Frequency analysis of noise segments in an audio file')
    parser.add_argument('input_file', action='store', type=str, help='input file to analyze')
    parser.add_argument('output_file', action='store', type=str, help='output file to write (default: stdout)', default=sys.stdout)
    parser.add_argument('-w, --window-size', action='store', type=int, dest='window_size', help='window size (default: 200ms)', default=200)
    parser.add_argument('-t, --noise-threshold', action='store', type=float, dest='noise_threshold', help='noise thresold (default: 0.25)', default=0.25)
    parser.add_argument('-m, --model-file', action='store', type=str, dest='model_file', help='the model file to use', default='models/params.pkl')
    args = parser.parse_args()

    # downsample file to 8KHz, 8 bits per sample
    in_file = args.input_file
    out_file = random_string() + ".wav"

    sox_in = pysox.CSoxStream(in_file)
    sox_out = pysox.CSoxStream(out_file, 'w', pysox.CSignalInfo(SAMPLE_RATE, 1, 8), fileType='wav')
    sox_chain = pysox.CEffectsChain(sox_in, sox_out)
    sox_chain.add_effect(pysox.CEffect("rate", [str(SAMPLE_RATE)]))
    sox_chain.flow_effects()
    sox_out.close()

    input_file = Sndfile(out_file, 'r')

    fs = input_file.samplerate
    num_frames = input_file.nframes

    window = args.window_size/1000.
    chunk_size = int(np.floor(window*fs))

    mlp = MLP_VAD(args.model_file)

    noise_seconds = []
    frame_num = 0
    frames_read = 0
    while (frames_read < num_frames):
        if frames_read + chunk_size > num_frames:
            break;

        signal = input_file.read_frames(chunk_size)
        frames_read = frames_read + chunk_size

        speech_prob = mlp.classify(fs, signal)
        print speech_prob

        if speech_prob <= args.noise_threshold:
            seconds = np.round(frame_num*window, 2)
            if len(noise_seconds) > 0 and (noise_seconds[-1][0] + noise_seconds[-1][1]) == seconds:
                noise_seconds[-1][1] += window
            else:
                noise_seconds.append([np.round(frame_num*window, 2), window])

        frame_num = frame_num + 1

    input_file.close()
    os.unlink(out_file)

    if len(noise_seconds) == 0:
        print "no noise segments found"
        sys.exit()

    # extract noise parts 
    kwargs = {'stdin': PIPE, 'stdout': PIPE, 'stderr': PIPE}
    sox_args = ['sox', args.input_file, args.output_file, 'remix', '-', 'trim']
    for i in noise_seconds:
        sox_args.append("=" + str(i[0]))
        sox_args.append(str(i[1]))

    pipe = Popen(sox_args, **kwargs)
    output, errors = pipe.communicate()

    if errors:
        raise RuntimeError(errors)
