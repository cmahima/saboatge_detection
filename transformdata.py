import  pandas as pd
import csv
import numpy as np

def get_dataframe(df):
    m= len(df.index)
    #label=[l]*m
    print(m)
    #df['label']=label
    index1= df[df['Elements'] == '/Marker/1'].index
    df = df.drop(range(0,index1[0]))

    index2= df[df['Elements'] == '/Marker/2'].index
    print(f"{index1[0]} {index2[0]}")

    df = df.drop(range(index2[0],len(df.index)))

    #df=df[["Delta_TP9","Delta_AF7","Delta_AF8","Delta_TP10","Theta_TP9","Theta_AF7","Theta_AF8","Theta_TP10","Alpha_TP9","Alpha_AF7","Alpha_AF8","Alpha_TP10","Beta_TP9","Beta_AF7","Beta_AF8","Beta_TP10","Gamma_TP9","Gamma_AF7","Gamma_AF8","Gamma_TP10"
#,"label"]]
    df=df[["RAW_TP9","RAW_AF7","RAW_AF8","RAW_TP10"]]
    df=df.dropna()
    df=df.reset_index()
    return df



def get_data(df,l):
    #df = df.iloc[:17000]
    df=get_dataframe(df)
    df=df.dropna()
    m= len(df.index)
    final_train_data=np.ndarray(shape=(int(m/31.5), 64, 4))
    final_trainy_data=np.ndarray(shape=int(m/31.5))

    win=64
    #feat=["Delta_TP9","Delta_AF7","Delta_AF8","Delta_TP10","Theta_TP9","Theta_AF7","Theta_AF8","Theta_TP10","Alpha_TP9","Alpha_AF7","Alpha_AF8","Alpha_TP10","Beta_TP9","Beta_AF7","Beta_AF8","Beta_TP10","Gamma_TP9","Gamma_AF7","Gamma_AF8","Gamma_TP10"]
    feat=["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]
    for i in range(len(feat)):
     k = 0
     df1 = df[feat[i]]
     m = len(df1.index)
     jj=0

     while ((k+1)*win<m):
        #filex='/Users/mahima/Desktop/Sab_data/Sergio/data_1_2_RAW128/train/data/'+feat[i]+'_train.txt'
        #filey='/Users/mahima/Desktop/Sab_data/Sergio/data_1_2_RAW128/train/'+feat[i]+'_y_train.txt'
        final_train_data[jj,:,i]=df1[int(k*win):int((k+1)*win)]
        final_trainy_data[jj]=l
        jj=jj+1
        k=k+0.5

    return final_train_data,final_trainy_data






