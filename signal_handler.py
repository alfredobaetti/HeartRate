import numpy as np
import scipy.fftpack as fftpack
from sklearn.decomposition import FastICA
import cv2 as cv


class Handler:
    def __init__(self, ROI):
        self.ROI = ROI
        

    def get_channel_signal(self):
        blue = []
        green = []
        red = []
        c=0

        b, g, r = cv.split(roi)

        g = np.mean(g)
        r = np.mean(r)
        b = np.mean(b)

        return g, r, b

        for roi in self.ROI:
            b, g, r = cv.split(roi)
            #b = np.mean(np.sum(b)) / np.std(b)
            #g = np.mean(np.sum(g)) / np.std(g)
            #r = np.mean(np.sum(r)) / np.std(r)
            if(c==1):
                g2 = np.mean(g)
                r2 = np.mean(r)
                b2 = np.mean(b)

                g=(g1+g2)/2
                r=(r1+r2)/2
                b=(b1+b2)/2

                blue.append(b)
                green.append(g)
                red.append(r)

                c=0

            else:
                g1 = np.mean(g)
                r1 = np.mean(r)
                b1 = np.mean(b)

                c=1

        return blue, green, red

    def ICA(self, matrix, n_component, max_iter=200):
        matrix = matrix.T
        ica = FastICA(n_components=n_component, max_iter=max_iter)
        u = ica.fit_transform(matrix)
        return u.T

    def fourier_transform(self, signal, N, fs):
        result = fftpack.fft(signal, N)
        result = np.abs(result)
        freqs = np.arange(N) / N
        freqs = freqs * fs
        return result[:N // 2], freqs[:N // 2]
