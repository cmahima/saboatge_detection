import utility
import numpy as np

def combine_data(*args):
    d = {}
    l=len(args)
    for i, val in enumerate(args):
        d["{}".format(i)] = np.asarray(val)

    totallen=sum((d["{}".format(i)].shape[0]) for i in range (1,l,2))
    minlen=min((d["{}".format(i)].shape[0]) for i in range (1,l,2))
    result_data=np.ndarray(shape=(int(minlen*(l/2)),63,63,4))
    result_data_y=np.ndarray(shape=(int(minlen*(l/2))))
    i=0
    j=0
    while(i<minlen):

        for j1 in range (0,l-1,2):
            result_data[j,:,:,:] =d["{}".format(j1)][i,:,:,:]
            result_data_y[j] =d["{}".format(j1+1)][i]
            j=j+1
        i=i+1

    total_len=result_data_y.shape[0]
    train_len=int(0.7*total_len)
    train_data=result_data[:train_len,:,:,:]
    train_data_y=result_data_y[:train_len]

    test_data=result_data[train_len:train_len+int(0.2*total_len),:,:,:]
    test_data_y=result_data_y[train_len:train_len+int(0.2*total_len)]

    valid_data=result_data[train_len+int(0.2*total_len):,:,:,:]
    valid_data_y=result_data_y[train_len+int(0.2*total_len):]

    return train_data,train_data_y,test_data,test_data_y,valid_data,valid_data_y


#combine_data ([[1,2,3,4],[7,3,5,5]],[1,2],[[8,5,3,4],[7,3,6,7],[1,3,8,5]],[7,3,5],[[3,2,3,43],[73,3,35,5]],[1,2])
