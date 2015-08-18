from mlp import MLP
from scipy.io import wavfile
import sys
import numpy as np
import theano
import theano.tensor as T

def main():
    fs, sig = wavfile.read(sys.argv[1])
    sig = np.array(sig.astype(np.float)/2**8, dtype=np.float)
    spec = np.abs(np.fft.fft(sig))

    data_x = np.zeros((1, 800))
    data_x[0] = spec
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=True)

    x = T.matrix('x')

    rng = np.random.RandomState(1234)

    classifier = MLP(
        rng=rng,
        input=x,
        n_in=800,
        n_hidden=500,
        n_out=2
    )

    classifier.load_model('params_acc_11p.pkl')

    index = T.lscalar()  # index to a [mini]batch

    predict_model = theano.function(
        inputs=[index],
        outputs=classifier.y_pred,
        givens={
            x: shared_x[index:index + 1],
        }
    )

    predicted_values = predict_model(0)
    labels=['noise', 'speech']
    print labels[predicted_values[0]]

if __name__ == '__main__':
    main()
