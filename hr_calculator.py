from scipy import signal
import matplotlib.pyplot as plt

def find_heart_rate(fft, freqs, freq_min, freq_max):
    fft_maximums = []

    for i in range(fft.shape[0]):
        if freq_min <= freqs[i] <= freq_max:
            fftMap = abs(fft[i])
            fft_maximums.append(fftMap.max())
        else:
            fft_maximums.append(0)

    plt.plot(fft_maximums, 'ro', linestyle = 'solid', color = 'red')
    plt.show()

    peaks, properties = signal.find_peaks(fft_maximums)
    max_peak = -1
    max_freq = 0

    # plt.plot(peaks, 'ro', linestyle = 'solid', color = 'blue')
    # plt.show()

    for peak in peaks:
        if fft_maximums[peak] > max_freq:
            max_freq = fft_maximums[peak]
            max_peak = peak

    plt.plot(freqs, 'ro', linestyle = 'solid', color = 'green')
    plt.show()

    return freqs[max_peak] * 60
