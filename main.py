from operator import le
from signal_handler import Handler
import numpy as np
import fft_filter
import dlib
from scipy import signal
import cv2 as cv
import hr_calculator
from sklearn.manifold import SpectralEmbedding
from MovingAverageFilter import MovingAverageFilter
import matplotlib.pyplot as plt
from imutils import face_utils
import scipy.fftpack as fftpack
from signal_processing import Signal_processing
import time
import read_json
import glob


freqs_min = 0.7
freqs_max = 2.7

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

def get_hr(green, fps):

    MovingAverage = MovingAverageFilter()

    t = np.arange(0, len(green)/fps, len(green)/fps/300)

    """GREEN CHANNEL"""
    signal = green
    #plt.title('Green channel')
    #plt.plot(t, signal, 'ro', linestyle = 'solid', color = 'green')
    #plt.show()

    """FILTRO DETREND"""
    signal = SignalProcessing.signal_detrending(signal)
    #plt.title('Filtro detrend')
    #plt.plot(t, signal, 'ro', linestyle = 'solid', color = 'green')
    #plt.show()

    """INTERPOLATION"""
    signal = SignalProcessing.interpolation(signal, times)
    #plt.title('Interpolación')
    #plt.plot(t, signal, 'ro', linestyle = 'solid', color = 'green')
    #plt.show()

    """NORMALIZACIÓN"""
    signal = SignalProcessing.normalization(signal)
    #plt.title('Normalización')
    #plt.plot(t, signal, 'ro', linestyle = 'solid', color = 'green')
    #plt.show()

    """FILTRO PARA SUAVIZAR CURVA"""
    for i in range(len(signal)):
        signal[i] = MovingAverage.start(signal[i])
    #plt.title('Filtro pasa-bajo (curva suavizada)')
    #plt.plot(t, signal, 'ro', linestyle = 'solid', color = 'green')
    #plt.show()

    fft, freqs = fft_filter.fft_filter(signal, freqs_min, freqs_max, fps)
    heartrate = hr_calculator.find_heart_rate(fft, freqs, freqs_min, freqs_max)

    return heartrate


def make_video():
    height, width, layers = frame.shape
    size = (width,height)
    out = cv.VideoWriter('video_10.mp4',cv.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()

if __name__ == '__main__':
    video_path = 'C:/Users/baett/OneDrive/Desktop/Dataset proyecto/videos proyecto/IMG_pei2.mp4'
    ROI = []
    HR = []
    BLUE = []
    GREEN = []
    RED = []
    frame_array = []
    heartrate = 0
    camera_code = 0
    capture = cv.VideoCapture(video_path)
    #fps = capture.get(cv.CAP_PROP_FPS)
    fps = 30
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    times = []
    t0 = time.time()
    SignalProcessing = Signal_processing()
    p=0
    cant_ROI = 0
    TSfr, TShr, HeartRate = read_json.getTS()
    #while capture.isOpened():
    for filename in glob.glob('C:/Users/baett/OneDrive/Desktop/Proyecto final/Dataset proyecto/04-01/*.png'):
        frame = cv.imread(filename)
    
        #ret, frame = capture.read()
        #if not ret:
            #continue

        grayf = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        face = detector(grayf, 0)
        
        if len(face) >0:
            times.append(time.time() - t0)
            shape = predictor(grayf, face[0])
            shape = shape_to_np(shape)  

            cv.rectangle(frame,(shape[54][0], shape[29][1]), #draw rectangle on right and left cheeks
                        (shape[12][0],shape[33][1]), (0,255,0), 0)
            cv.rectangle(frame, (shape[4][0], shape[29][1]), 
                    (shape[48][0],shape[33][1]), (0,255,0), 0)  
            ROI1 = frame[shape[29][1]:shape[33][1], #right cheek
                    shape[54][0]:shape[12][0]]  
            ROI2 =  frame[shape[29][1]:shape[33][1], #left cheek
                    shape[4][0]:shape[48][0]]   

            #for (x, y) in shape:
            #    cv.circle(frame, (x, y), 1, (0, 0, 255), -1) #draw facial landmarks

            b1, g1, r1 = Signal_processing.get_channel_signal(ROI1)
            b2, g2, r2 = Signal_processing.get_channel_signal(ROI2)

            g = (g1+g2)/2

            GREEN.append(g)

        if len(GREEN) == 300:
            heartrate = get_hr(GREEN, fps)
            HR.append(heartrate)

            for i in range(30):
                GREEN.pop(0)
                times.pop(0)

        if len(HR) > 10:
            cv.putText(frame, '{:.0f}bpm'.format(np.mean(HR)), (200, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            try:    
                cv.putText(frame, '{:.0f}bpm'.format(HeartRate[p]), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
            except:
                continue

        if len(HR) > 20:
            for i in range(2):
                HR.pop(0)

        print(np.mean(HR))
        p=p+1
        cv.imshow('frame', frame)
        frame_array.append(frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    make_video()
