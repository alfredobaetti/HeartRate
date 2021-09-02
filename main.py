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

freqs_min = 0.7
freqs_max = 4

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

def get_hr(ROI, fps):

    signal_handler = Handler(ROI)
    MovingAverage = MovingAverageFilter(1)
    SignalProcessing = Signal_processing()

    """Método GUI"""
    # g = SignalProcessing.extract_color(ROI)

    # detrended_data = SignalProcessing.signal_detrending(g)

    # #interpolated_data = SignalProcessing.interpolation(detrended_data, times)

    # normalized_data = SignalProcessing.normalization(detrended_data)

    # fft_of_interest, freqs_of_interest = SignalProcessing.fft(normalized_data, fps)
    
    # max_arg = np.argmax(fft_of_interest)
    # heartrate_1 = freqs_of_interest[max_arg]



    """ICA"""
    # blue, green, red = signal_handler.get_channel_signal()
    # matrix = np.array([blue, green, red])
    # component = signal_handler.ICA(matrix, 3)
    # le_X = component[1]


    """GREEN CHANNEL"""
    blue, green, red = signal_handler.get_channel_signal()
    green = np.array(green)
    le_X = green


    """LAPLACIAN EIGENMAPS"""
    # blue, green, red = signal_handler.get_channel_signal()
    # matrix = np.vstack((blue,green,red))
    # matrix = matrix.T
    # le = SpectralEmbedding(n_components=1)
    # le_X = le.fit_transform(matrix)


    plt.plot(le_X, 'ro', linestyle = 'solid', color = 'green')
    plt.show()

    """FILTRO PARA ELIMINAR PUNTOS SINGULARES"""
    # for i in range(len(le_X)):
    #     if(abs(le_X[i]-np.mean(le_X))>10*np.mean(le_X)):
    #         le_X[i] = le_X[i-1]

    """FILTRO DETREND"""
    le_X = signal.detrend(le_X)

    plt.plot(le_X, 'ro', linestyle = 'solid', color = 'red')
    plt.show()

    """FILTRO PARA SUAVIZAR CURVA"""
    for i in range(len(le_X)):
        le_X[i] = MovingAverage.start(le_X[i])

    plt.plot(le_X[7:], 'ro', linestyle = 'solid', color = 'blue')
    plt.show()


    """BANDPASS FILTER"""
    # le_X = fft_filter.butter_bandpass_filter(data=le_X[7:],lowcut=0.7,highcut=3,fs=fps,order = 5)

    # plt.plot(le_X, 'ro', linestyle = 'solid', color = 'red')
    # plt.show()


    """POWER SPECTRAL DENSITY"""
    # ps = np.abs(np.fft.fft(le_X))**2
    # freqs = np.fft.fftfreq(le_X.shape[0], d=1.0 / fps)
    # 
    # idx2 = np.argmax(ps)

    #fft, freqs = signal.welch(le_X, fs=fps)

    # plt.plot(ps, 'ro', linestyle = 'solid', color = 'red')
    # plt.show()

    # plt.plot(freqs, 'ro', linestyle = 'solid', color = 'blue')
    # plt.show()

    #idx2 = np.argmax(ps)

    #heartrate_1 = freqs[idx2] * 60
    #component = signal_handler.ICA(matrix, 3)
    fft, freqs = fft_filter.fft_filter(le_X, freqs_min, freqs_max, fps)
    heartrate_1 = hr_calculator.find_heart_rate(fft, freqs, freqs_min, freqs_max)
    # fft, freqs = fft_filter.fft_filter(component[1], freqs_min, freqs_max, fps)
    # heartrate_2 = hr_calculator.find_heart_rate(fft, freqs, freqs_min, freqs_max)
    # fft, freqs = fft_filter.fft_filter(component[2], freqs_min, freqs_max, fps)
    # heartrate_3 = hr_calculator.find_heart_rate(fft, freqs, freqs_min, freqs_max)
    #return (heartrate_1 + heartrate_2 + heartrate_3) / 3
    return heartrate_1



if __name__ == '__main__':
    video_path = 'IMG_1603.mp4'
    ROI = []
    heartrate = 0
    camera_code = 0
    capture = cv.VideoCapture(video_path)
    fps = capture.get(cv.CAP_PROP_FPS)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            continue
        #dects = detector(frame)
        grayf = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        face = detector(grayf, 0)
        #for face in dects:
        if len(face) >0:
            shape = predictor(grayf, face[0])
            shape = shape_to_np(shape)  
            # for (a, b) in shape:
            #      cv.circle(frame, (a, b), 1, (0, 0, 255), -1) #draw facial landmarks    
            left = face[0].left()
            right = face[0].right()
            top = face[0].top()
            bottom = face[0].bottom()
            h = bottom - top
            w = right - left    
            #roi = frame[top + h // 10 * 2:top + h // 10 * 7, left + w // 9 * 2:left + w // 9 * 8]  
            cv.rectangle(frame,(shape[54][0], shape[29][1]), #draw rectangle on right and left cheeks
                        (shape[12][0],shape[33][1]), (0,255,0), 0)
            cv.rectangle(frame, (shape[4][0], shape[29][1]), 
                    (shape[48][0],shape[33][1]), (0,255,0), 0)  
            ROI1 = frame[shape[29][1]:shape[33][1], #right cheek
                    shape[54][0]:shape[12][0]]  
            ROI2 =  frame[shape[29][1]:shape[33][1], #left cheek
                    shape[4][0]:shape[48][0]]   
            #cv.rectangle(frame, (left + w // 9 * 2, top + h // 10 * 3), (left + w // 9 * 8, top + h // 10 * 7), color=(0, 0, 255))
            #cv.rectangle(frame, (left, top), (left + w, top + h), color=(0, 0, 255))
            #ROI3 = (ROI1 + ROI2) /2
            ROI.append(ROI1)    
            #ROI.append(ROI2)
        # left = face.left()
        # right = face.right()
        # top = face.top()
        # bottom = face.bottom()
        # h = bottom - top
        # w = right - left
        # roi = frame[top + h // 10 * 2:top + h // 10 * 7, left + w // 9 * 2:left + w // 9 * 8]
        # cv.rectangle(frame, (left + w // 9 * 2, top + h // 10 * 3), (left + w // 9 * 8, top + h // 10 * 7),
        #              color=(0, 0, 255))
        # cv.rectangle(frame, (left, top), (left + w, top + h), color=(0, 0, 255))
        # ROI.append(roi)
        if len(ROI) == 100:
            heartrate = get_hr(ROI, fps)
            for i in range(10):
                ROI.pop(0)

        cv.putText(frame, '{:.1f}bps'.format(heartrate), (50, 300), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
