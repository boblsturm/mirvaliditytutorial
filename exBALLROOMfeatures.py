# get paths and labels to each recording
import glob
import pandas as pd
import numpy as np

dd = {}
counter = 0
for ff in glob.iglob('/Users/bobs/research/datasets/extendedballroom_v1.1/*/*.mp3'):
    dd[counter] = {'path':ff, 'class':ff.split('/')[6]}
    counter+=1

# create pandas dataframe
wav_df_ext = pd.DataFrame.from_dict(dd,orient='index')
print(wav_df_ext)

import librosa
from statsmodels.tsa.ar_model import AutoReg

def extractFeatures(df,hop_length=1024,ARorder=12,samplerate=22050,minlag=5,maxlag=90):
    ARparams = []
    for index, row in df.iterrows():
        print(row['path'])
        y, sr = librosa.load(row['path'])
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

ARparams = extractFeatures(wav_df_ext)
wav_df_ext = pd.concat([wav_df_ext, pd.DataFrame(ARparams)], axis=1)
print(wav_df_ext)
wav_df_ext.to_pickle('extendedBALLROOMfeatures.pkl')
