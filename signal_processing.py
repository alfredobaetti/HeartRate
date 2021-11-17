import cv2
import numpy as np
import time
from scipy import signal


class Signal_processing():
    def __init__(self):
        self.a = 1
    
    def get_channel_signal(ROI):
        # blue = []
        # green = []
        # red = []

        b, g, r = cv2.split(ROI)

        g = np.mean(g)
        r = np.mean(r)
        b = np.mean(b)

        return b, g, r

    def normalization(self, data_buffer):
        '''
        normalize the input data buffer
        '''
        
        #normalized_data = (data_buffer - np.mean(data_buffer))/np.std(data_buffer)
        normalized_data = data_buffer/np.linalg.norm(data_buffer)
        
        return normalized_data
    
    def signal_detrending(self, data_buffer):
        '''
        remove overall trending
        
        '''
        detrended_data = signal.detrend(data_buffer)
        
        return detrended_data
        
    def interpolation(self, data_buffer, times):
        '''
        interpolation data buffer to make the signal become more periodic (advoid spectral leakage)
        '''
        L = len(data_buffer)
        
        even_times = np.linspace(times[0], times[-1], L)
        
        interp = np.interp(even_times, times, data_buffer[:])
        interpolated_data = np.hamming(L) * interp
        return interpolated_data
        
    
        
    
        
        
        
        
        
        
        
        