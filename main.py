import numpy as np
import io
import pyedflib
import scipy.io as sio
import os


def load_edf_signals(path):
    sig = pyedflib.EdfReader(path)
    n = sig.signals_in_file
    sigbuf = np.zeros((n, sig.getNSamples()[0]))

    for i in range(n):
        sigbuf[i, :] = sig.readSignal(i)
    annotations = sig.read_annotation()

    return sigbuf.transpose(), annotations


sig, annotation = load_edf_signals('./datasets/S001R04.edf')
print(sig.shape)
print(annotation)