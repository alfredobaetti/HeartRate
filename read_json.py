import json
import numpy
import numpy as np
import math
import pickle

def find_nearest(array,value):  
    
        idx = np.searchsorted(array, value, side="left")
        
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return array[idx-1], idx
        
        else:
            return array[idx], idx

def getTS():
    with open("C:/Users/baett/OneDrive/Desktop/Proyecto final/Dataset proyecto/04-01.json") as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()

    HR = []
    TShr = []
    TSfr = []
    TShraprox = []
    HRaprox = []

    for c in range (len(jsonObject['/FullPackage'])):
        pulserate = jsonObject['/FullPackage'][c]['Value']['pulseRate']
        tsheartrate = jsonObject['/FullPackage'][c]['Timestamp']
        HR.append(pulserate)
        TShr.append(tsheartrate)

    for c in range (len(jsonObject['/Image'])):
        tsframe = jsonObject['/Image'][c]['Timestamp']
        TSfr.append(tsframe)

    for c in TSfr:
        ts, idx = find_nearest(TShr, c)
        TShraprox.append(ts)
        try:
            HRaprox.append(HR[idx])
        except:
            pass
    
    return TSfr, TShraprox, HRaprox

