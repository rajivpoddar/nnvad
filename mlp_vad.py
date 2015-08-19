from mlp import MLP
from scipy.io import wavfile
import sys
import numpy as np
import theano
import theano.tensor as T

def main():
    fs, sig = wavfile.read(sys.argv[1])
    sig = np.asarray(sig, dtype=np.float)/2**8
    sig = sig.reshape((8, 200))
    spec = np.abs(np.fft.fft(sig))

    shared_x = theano.shared(np.asarray(spec, dtype=theano.config.floatX), borrow=True)

    x = T.matrix('x')

    rng = np.random.RandomState(1234)

    classifier = MLP(
        rng=rng,
        input=x,
        n_in=800,
        n_hidden=500,
        n_out=2
    )

    classifier.load_model('params_acc_8p.pkl')

    index = T.lscalar()  # index to a [mini]batch

    predict_model = theano.function(
        inputs=[index],
        outputs=classifier.y_pred,
        givens={
            x: shared_x[index:index + 1],
        }
    )

    predicted_values = [predict_model(i)[0] for i in xrange(len(sig))]
    mean = np.round(np.mean(predicted_values), 2)
    if mean < 0.25:
        print "noise (%.2f)" % (mean)
    else:
        print "speech (%.2f)" % (mean)

if __name__ == '__main__':
    main()
