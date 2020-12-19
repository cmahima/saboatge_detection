import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
import math
from scipy.stats.stats import pearsonr
import eeglib
from scipy.stats import kurtosis, skew
import requests
from csv import writer
from scipy import stats
import os
import time
start_time = time.time()
para = 0
f = open('test.csv', "w+")
f.close()
f = open('test.csv', "w+")
f.close()
df1 = pd.read_csv("realtimeeeg.csv")
baseidx = np.zeros(2)
baseidx[0] = 125
baseidx[1] = 256
df = df1[0:256]


# continuous wavelet transform
# wavelet parameters
[m, n] = df.shape
num_frex = 30
min_freq = 8
max_freq = 27
srate = 256
frex = np.linspace(min_freq, max_freq, num_frex)
time1 = np.arange(-1.5, 1.5, 1 / srate)
half_wave = round((len(time1) - 1) / 2)
# FFT parameters
nKern = len(time1)
nData = m
nConv = nKern + nData - 1
# initialize output time-frequency data

# baseidx[0] = find_nearest(df['time'], baseline_window[0])
# baseidx[1] = find_nearest(df['time'], baseline_window[1])
baseidx = baseidx.astype(int)
tf = [[[0] * m] * len(frex)] * 4
tf = np.asarray(tf, dtype=float)
channels = ['RAW_TP10', 'RAW_AF7', 'RAW_AF8', 'RAW_TP9']
for cyclei in range(0, 4):
    dataX = fft(df[channels[cyclei]].to_numpy(), nConv)
    for i in range(0, len(frex)):
        s = 8 / (2 * math.pi * frex[i])
        cmw = np.multiply(np.exp(np.multiply(2 * complex(0, 1) * math.pi * frex[i], time1)),
                          np.exp(np.divide(-time1 ** 2, (2 * s ** 2))))
        cmwX = fft(cmw, nConv)
        cmwX = np.divide(cmwX, max(cmwX))
        as1 = ifft(np.multiply(cmwX, dataX), nConv)
        as1 = as1[half_wave:len(as1) - half_wave + 1]
        as1 = np.reshape(as1, m);
        mag = np.absolute(as1) ** 2
        tf[cyclei, i, :] = np.absolute(as1) ** 2;
    # print((np.squeeze(tf[cyclei,:,:])/ np.transpose(np.mean(tf[cyclei,:,baseidx[0]:baseidx[1]],1))))
    var = np.transpose(np.mean(tf[cyclei, :, baseidx[0]:baseidx[1]], 1))
rows = len(df1.index)

while (1):
    if (df.empty):
        print("No data present because of no connection")
        end= time.time()
        print( end - start_time)
        break
    if (rows > 256):
        df = df1[para * 26:(para + 1) * 26]
        # continuous wavelet transform
        # wavelet parameters
        [m, n] = df.shape
        num_frex = 30
        min_freq = 8
        max_freq = 27
        srate = 256
        frex = np.linspace(min_freq, max_freq, num_frex)
        time1 = np.arange(-1.5, 1.5, 1 / srate)
        half_wave = round((len(time1) - 1) / 2)
        # FFT parameters
        nKern = len(time1)
        nData = m
        nConv = nKern + nData - 1
        # initialize output time-frequency data

        # baseidx[0] = find_nearest(df['time'], baseline_window[0])
        # baseidx[1] = find_nearest(df['time'], baseline_window[1])
        baseidx = baseidx.astype(int)
        tf = [[[0] * m] * len(frex)] * 4
        tf = np.asarray(tf, dtype=float)
        channels = ['RAW_TP10', 'RAW_AF7', 'RAW_AF8', 'RAW_TP9']
        for cyclei in range(0, 4):
            dataX = fft(df[channels[cyclei]].to_numpy(), nConv)
            for i in range(0, len(frex)):
                s = 8 / (2 * math.pi * frex[i])
                cmw = np.multiply(np.exp(np.multiply(2 * complex(0, 1) * math.pi * frex[i], time1)),
                                  np.exp(np.divide(-time1 ** 2, (2 * s ** 2))))
                cmwX = fft(cmw, nConv)
                cmwX = np.divide(cmwX, max(cmwX))
                as1 = ifft(np.multiply(cmwX, dataX), nConv)
                as1 = as1[half_wave:len(as1) - half_wave + 1]
                as1 = np.reshape(as1, m);
                mag = np.absolute(as1) ** 2
                tf[cyclei, i, :] = np.absolute(as1) ** 2;
            # print((np.squeeze(tf[cyclei,:,:])/ np.transpose(np.mean(tf[cyclei,:,baseidx[0]:baseidx[1]],1))))
            # var = np.transpose(np.mean(tf[cyclei, :, baseidx[0]:baseidx[1]], 1))
            # print(var)

            tf[cyclei, :, :] = 10 * np.log10(np.divide((np.squeeze(tf[cyclei, :, :])).T, var).T)

        # tf=tf[:,1:len(frex),:]
        #        pts = np.where(np.logical_and(df['time'] >= 0.0, df['time'] <= 4.0))
        #       pts = np.asarray(pts)
        #      [m1, n1] = pts.shape
        #      tf = tf[:, :, pts]
        #     tf = np.reshape(tf, (4, len(frex), n1))

        falpha = np.where(np.logical_and(frex >= 8, frex <= 13))
        falpha = np.asarray(falpha)
        tfalpha = [[0] * m] * 4
        tfalpha = np.array(tfalpha, dtype=float)
        for i in range(4):
            tfalpha[i] = np.mean(tf[i, falpha, :], axis=1)

        fbeta = np.where(np.logical_and(frex >= 13, frex <= 27))
        fbeta = np.asarray(fbeta)
        tfbeta = [[0] * m] * 4
        tfbeta = np.array(tfbeta, dtype=float)
        for i in range(4):
            tfbeta[i] = np.mean(tf[i, fbeta, :], axis=1)

        '''
        #lines 114-180 extract 78 features from the
        #EEG data
        '''
        features = np.zeros(shape=78)
        mobilityalpha = np.zeros(shape=4)
        mobilitybeta = np.zeros(shape=4)
        j = 0

        for i in range(4):
            features[j] = np.mean(tfalpha[i, :])
            j = j + 1

        for i in range(4):
            features[j] = np.mean(tfbeta[i, :])
            j = j + 1

        for i in range(4):
            features[j] = np.var(tfalpha[i, :])
            j = j + 1

        for i in range(4):
            features[j] = np.var(tfbeta[i, :])
            j = j + 1

        features[j] = features[0] + features[2] - (features[1] + features[3])
        j = j + 1
        features[j] = features[4] + features[6] - (features[5] + features[7])
        j = j + 1

        for i in range(4):
            for l in range(4):
                if i > l:
                    features[j] = pearsonr(np.transpose(tfbeta[i, :]), np.transpose(tfbeta[l, :]))[0]
                    j = j + 1
                    features[j] = pearsonr(np.transpose(tfalpha[i, :]), np.transpose(tfalpha[l, :]))[0]
                    j = j + 1
                    features[j] = pearsonr(np.transpose(tfbeta[i, :]), np.transpose(tfalpha[l, :]))[0]
                    j = j + 1
                    features[j] = pearsonr(np.transpose(tfbeta[l, :]), np.transpose(tfalpha[i, :]))[0]
                    j = j + 1
                if i == l:
                    features[j] = pearsonr(np.transpose(tfbeta[i, :]), np.transpose(tfalpha[l, :]))[0]
                    j = j + 1

        for i in range(4):
            features[j] = eeglib.features.hjorthMobility(tfalpha[i, :])
            j = j + 1

        for i in range(4):
            features[j] = eeglib.features.hjorthMobility(tfbeta[i, :])
            j = j + 1

        for i in range(4):
            features[j] = eeglib.features.hjorthComplexity(tfalpha[i, :])
            j = j + 1

        for i in range(4):
            features[j] = eeglib.features.hjorthComplexity(tfbeta[i, :])
            j = j + 1

        for i in range(4):
            features[j] = skew(tfbeta[i, :])
            j = j + 1

        for i in range(4):
            features[j] = skew(tfalpha[i, :])
            j = j + 1

        for i in range(4):
            features[j] = kurtosis(tfalpha[i, :])
            j = j + 1

        for i in range(4):
            features[j] = kurtosis(tfbeta[i, :])
            j = j + 1
        para += 1

        with open(r'/Users/mahima/Downloads/mlmodelmuse/test.csv', 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(features)
    if (para >=6):
        break


if (os.stat("test.csv").st_size == 0):
    print("Connection not made with the device")
    print( time.time() - start_time)
else:
    df = pd.read_csv('test.csv',header=None)
    if (len(df.index)==1):
        df.to_csv(r'test2.csv', index=False, header=None)
    elif(len(df.index)>1):
        df1 = stats.zscore(df)
        df1 = pd.DataFrame(data=df1)
        df1.to_csv(r'test2.csv', index=False, header=None)
    colors = []
    file1 = open('test2.csv', 'r')
    Lines = file1.readlines()
    for line in Lines:
        line = line.strip('\n')
        data = "[" + line + "]"
        # data=line
        url = 'http://ec2-100-25-104-74.compute-1.amazonaws.com/predict'
        headers = {
            'Content-type': 'application/json',
        }
        f = requests.post(url, headers=headers, data=data)
        text = f.text
        text = text.rstrip(' \x1b[0m\n')
        text = text.split()[-1]
        colors.append(text)
    color = max(colors, key=colors.count)
    print(color)

    end = time.time()
    print(end - start_time)
