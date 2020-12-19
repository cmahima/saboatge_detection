import data
import tfanalysis
from sklearn import preprocessing
import pandas as pd
from scipy import stats


def getdata(df,label):
    df_fin=data.get_data(df,label)
    label=df_fin['label']
    cols=["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]
    x = df_fin[["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]].values
    min_max_scaler =preprocessing.MinMaxScaler(feature_range=(0, 1))
    #scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
   # x_scaled = stats.zscore(x)
    df_1 = pd.DataFrame(x_scaled,columns=cols)
    df_1['label']=label

    pre_data,labels=tfanalysis.tf(df_fin)
    return pre_data,labels

