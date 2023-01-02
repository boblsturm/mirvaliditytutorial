#!/usr/bin/python3

import glob
import pandas as pd
import numpy as np
import rubberband
import librosa
from statsmodels.tsa.ar_model import AutoReg

dd = {}
counter = 0
for ff in glob.iglob('BALLROOM/*/*.wav'):
    dd[counter] = {'path':ff, 'class':ff.split('/')[1]}
    counter+=1

wav_df = pd.DataFrame.from_dict(dd,orient='index')

def extractFeaturesDilatedAudio(df,Dfactor,hop_length=1024,ARorder=12,samplerate=22050,minlag=5,maxlag=90):
    ARparams = []
    for index, row in df.iterrows():
        print(Dfactor,row['path'])
        y, sr = librosa.load(row['path'])
        y = rubberband.stretch(y,rate=sr,ratio=Dfactor)
        if sr != samplerate:
            y = librosa.resample(y, sr, samplerate)
        oenv = librosa.onset.onset_strength(y=y, sr=samplerate, aggregate=np.median, \
                                            hop_length=hop_length)
        ac_global = librosa.autocorrelate(oenv)
        ac_global = librosa.util.normalize(ac_global)
        ac_global = ac_global[minlag:maxlag]
        ac_global = ac_global - np.mean(ac_global)
        ARmodel = AutoReg(ac_global,ARorder)
        ARmodel_fit = ARmodel.fit() 
        ARparams.append(ARmodel_fit._params)
    return ARparams

#for dfactor in [-15, -10, -8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8, 10, 15]:
for dfactor in [10, 15]:
    ARparams = extractFeaturesDilatedAudio(wav_df,1+dfactor/100.0)
    wav_df2 = pd.concat([wav_df, pd.DataFrame(ARparams)], axis=1)
    wav_df2.to_pickle('BALLROOMfeatures_dfactor'+str(dfactor)+'.pkl')
