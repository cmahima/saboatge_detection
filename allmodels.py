import pywt
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import History
history = History()
from sklearn.preprocessing import StandardScaler
from keras.layers import ConvLSTM2D
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras_self_attention import SeqSelfAttention
import data
import itertools
import tfanalysis
import pandas as pd
import statistics

import utility
import format_data
from sklearn import preprocessing
import keras_metrics
import transformdata
import combine2
from keras.layers import Bidirectional
from keras.layers import LSTM
from sklearn.metrics import confusion_matrix


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels
cm_plot_labels = ['no_sabotage','sabotage']
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

df_1=pd.read_csv("/Users/mahima/Downloads/sergio_1.csv")
df_2=pd.read_csv("/Users/mahima/Downloads/sergio_2.csv")
#df_3=pd.read_csv("/Users/mahima/Downloads/sergio_3.csv")
#df_4=pd.read_csv("/Users/mahima/Downloads/sergio_4.csv")
#df_5=pd.read_csv("/Users/mahima/Downloads/mohita_1.csv")
#df_6=pd.read_csv("/Users/mahima/Downloads/mohita_2.csv")
#df_7=pd.read_csv("/Users/mahima/Downloads/mohita_3.csv")
#df_8=pd.read_csv("/Users/mahima/Downloads/mohita_4.csv")
#df_9=pd.read_csv("/Users/mahima/Downloads/mahima_1.csv")
#df_10=pd.read_csv("/Users/mahima/Downloads/mahima_2.csv")
#df_11=pd.read_csv("/Users/mahima/Downloads/mahima_3.csv")
#df_12=pd.read_csv("/Users/mahima/Downloads/mahima_4.csv")
#df_13=pd.read_csv("/Users/mahima/Downloads/karan_1.csv")
#df_14=pd.read_csv("/Users/mahima/Downloads/karan_2.csv")
#df_15=pd.read_csv("/Users/mahima/Downloads/karan_3.csv")
#df_16=pd.read_csv("/Users/mahima/Downloads/karan_4.csv")

df_1,label_1=transformdata.get_data(df_1,0)
df_2,label_2=transformdata.get_data(df_2,1)
#df_3,label_3=transformdata.get_data(df_3,1)
#df_4,label_4=transformdata.get_data(df_4,1)
#df_5,label_5=transformdata.get_data(df_5,0)
#df_6,label_6=transformdata.get_data(df_6,0)
#df_7,label_7=transformdata.get_data(df_7,1)
#df_8,label_8=transformdata.get_data(df_8,1)
#df_9,label_9=transformdata.get_data(df_9,0)
#df_10,label_10=transformdata.get_data(df_10,0)
#df_11,label_11=transformdata.get_data(df_11,1)
#df_12,label_12=transformdata.get_data(df_12,1)
#df_13,label_13=transformdata.get_data(df_13,0)
#df_14,label_14=transformdata.get_data(df_14,0)
#df_15,label_15=transformdata.get_data(df_15,1)
#df_16,label_16=transformdata.get_data(df_16,1)

scales = range(1,64)
test_score=[]
train_data_cwt_1,train_labels_1,test_data_cwt,test_labels,valid_data_cwt,valid_labels=combine2.combine_data(df_1,label_1,df_2,label_2)#,df_3,label_3,df_4,label_4,df_5,label_5,df_6,label_6,df_7,label_7,df_8,label_8,df_9,label_9,df_10,label_10,df_11,label_11,df_12,label_12,df_13,label_13,df_14,label_14,df_15,label_15,df_16,label_16)
print("*********************************************************")
#df_1=pd.read_csv("/Users/mahima/Downloads/sergio_1.csv")
#df_2=pd.read_csv("/Users/mahima/Downloads/sergio_2.csv")
#df_3=pd.read_csv("/Users/mahima/Downloads/sergio_3.csv")
#df_4=pd.read_csv("/Users/mahima/Downloads/sergio_4.csv")
#df_1,label_1=transformdata.get_data(df_1,0)
#df_2,label_2=transformdata.get_data(df_2,0)
#df_3,label_3=transformdata.get_data(df_3,1)
#df_4,label_4=transformdata.get_data(df_4,1)
#test_data_cwt,test_labels,test_data_cwt_1,test_labels_1,valid_data_cwt,valid_labels=combine2.combine_data(df_1,label_1,df_2,label_2,df_3,label_3,df_4,label_4)

scalers = {}
for i in range(train_data_cwt_1.shape[1]):
    scalers[i] = StandardScaler()
    train_data_cwt_1[:, i, :] = scalers[i].fit_transform(train_data_cwt_1[:, i, :])

for i in range(test_data_cwt.shape[1]):
    test_data_cwt[:, i, :] = scalers[i].transform(test_data_cwt[:, i, :])

#for i in range(valid_data_cwt.shape[1]):
    #valid_data_cwt[:, i, :] = scalers[i].transform(valid_data_cwt[:, i, :])
[no_signals_train, no_steps_train, no_components_train] = np.shape(train_data_cwt_1)
[no_signals_test, no_steps_test, no_components_test] = np.shape(test_data_cwt)
[no_signals_valid, no_steps_valid, no_components_valid] = np.shape(valid_data_cwt)

train_data_cwt,train_labels=randomize(train_data_cwt_1,train_labels_1)
test_data_cwt,test_labels=randomize(test_data_cwt,test_labels)
valid_data_cwt,valid_labels=randomize(valid_data_cwt,valid_labels)
scales = range(1,64)
waveletname = 'morl'
train_size = no_signals_train
train_data_cwt_new = np.ndarray(shape=(train_size, 63, 63, 4))

for ii in range(0,train_size):
    if ii % 100 == 0:
        print(ii)
    for jj in range(0,4):
        signal = train_data_cwt[ii, :, jj]
        coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        coeff_ = coeff[:,:63]
        train_data_cwt_new[ii, :, :, jj] = coeff_

test_size = no_signals_test
test_data_cwt_new = np.ndarray(shape=(test_size, 63, 63, 4))
for ii in range(0,test_size):
    if ii % 100 == 0:
        print(ii)
    for jj in range(0,4):
        signal = test_data_cwt[ii, :, jj]
        coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        coeff_ = coeff[:,:63]
        test_data_cwt_new[ii, :, :, jj] = coeff_

valid_size = no_signals_valid
valid_data_cwt_new = np.ndarray(shape=(valid_size, 63, 63, 4))
for ii in range(0,valid_size):
    if ii % 100 == 0:
        print(ii)
    for jj in range(0,4):
        signal = valid_data_cwt[ii, :, jj]
        coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        coeff_ = coeff[:,:63]
        valid_data_cwt_new[ii, :, :, jj] = coeff_

x_train = train_data_cwt_new
y_train = train_labels
x_test = test_data_cwt_new
y_test = test_labels
x_valid = valid_data_cwt_new
y_valid = valid_labels
img_x = 63
img_y = 63
img_z = 4
num_classes = 2

batch_size = 16
epochs =60

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
input_shape = (img_x, img_y,img_z)

# convert the data to the right type
#x_train = x_train.reshape(x_train.shape[0],img_z, img_x, img_y, 1)
#x_test = x_test.reshape(x_test.shape[0], img_z,img_x, img_y, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[0]
n_steps, n_length = 2, 32
#trainX = x_train.reshape((x_train.shape[0], n_steps, n_length, n_features))
#testX = x_test.reshape((x_test.shape[0], n_steps, n_length, n_features))
#validX=x_valid.reshape((x_test.shape[0], n_steps, n_length, n_features))
# convert the data to the right type
#x_train = x_train.reshape(x_train.shape[0],img_z, img_x, img_y, 1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
yyy=x_test[0:400,:,:]
input_shape = (img_x, img_y,img_z)

# convert the data to the right type
#x_train = x_train.reshape(x_train.shape[0],img_z, img_x, img_y, 1)
#x_test = x_test.reshape(x_test.shape[0], img_z,img_x, img_y, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
print("-------CNN LSTM----------")
y_valid = keras.utils.to_categorical(y_valid, num_classes)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
model.add((Conv2D(filters=32, kernel_size=3, activation='relu')))
print(model.output_shape)

model.add((Dropout(0.5)))
model.add((MaxPooling2D(pool_size=2)))
model.add((Flatten()))
model.add(keras.layers.Reshape((32,-1)))

print(model.output_shape)
model.add(LSTM(100,return_sequences=True)
)
#model.add(LSTM(100,return_sequences=True)
#)
print(model.output_shape)
print("Hiiiii",model.input_shape)

model.add(SeqSelfAttention(attention_activation='sigmoid'))
print(f"{model.output_shape} is attention")

model.add(keras.layers.Reshape((32,100)))
print(f"{model.output_shape} is reshape")
model.add(Flatten())
model.add(Dropout(0.5))
print(f"{model.output_shape} is flatten")

model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy',keras_metrics.precision(),keras_metrics.recall(), keras_metrics.f1_score(),keras_metrics.true_positive(),keras_metrics.true_negative(),keras_metrics.false_positive(),keras_metrics.false_negative()])


history1=model.fit(x_train, y_train, batch_size=batch_size, validation_split=0.2,
          epochs=epochs, verbose=1)
model.summary()
train_score = model.evaluate(x_train, y_train, verbose=0)
print('train loss: {}, train accuracy: {},train precision: {},train recall: {},train fmeasure: {},train TP: {},train TN: {},train FP: {},train FN: {},'.format(train_score[0], train_score[1],train_score[2],train_score[3],train_score[4],train_score[5],train_score[6],train_score[7],train_score[8]))
test_score = model.evaluate(x_test, y_test, verbose=0)
print('test loss: {}, test accuracy: {},test precision: {},test recall: {},test fmeasure: {},test TP: {},test TN: {},test FP: {},test FN: {},'.format(test_score[0], test_score[1],test_score[2],test_score[3],test_score[4],test_score[5],test_score[6],test_score[7],test_score[8]))


print("----------------CONVLSTM---------------------")
input_shape = (1,img_x, img_y, img_z)
# convert the data to the right type
x_train = x_train.reshape(x_train.shape[0], 1,img_x, img_y, img_z)
x_test = x_test.reshape(x_test.shape[0],1, img_x, img_y, img_z)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
model = Sequential()
model.add(ConvLSTM2D(filters=32,  kernel_size=(2, 2), strides=(1, 1), activation='relu', input_shape=(input_shape),return_sequences=True))
model.add(keras.layers.BatchNormalization())
model.add(ConvLSTM2D(filters=32,  kernel_size=(2, 2), strides=(1, 1), activation='relu', input_shape=(input_shape),return_sequences=True))
model.add(keras.layers.BatchNormalization())
print(model.output_shape)
model.add(keras.layers.Reshape((32,-1)))
print(model.output_shape)
model.add(SeqSelfAttention(attention_activation='sigmoid'))
print(model.output_shape)
model.add(keras.layers.Reshape((61,61,32)))
#model.add(keras.layers.Reshape((30,30,32)))
print(model.output_shape)
model.add(Dropout(0.5))
model.add(Flatten())
print(model.output_shape)
model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy',keras_metrics.precision(),keras_metrics.recall(), keras_metrics.f1_score(),keras_metrics.true_positive(),keras_metrics.true_negative(),keras_metrics.false_positive(),keras_metrics.false_negative()])


history2=model.fit(x_train, y_train, batch_size=batch_size, validation_split=0.2,
          epochs=epochs, verbose=1)
model.summary()
train_score = model.evaluate(x_train, y_train, verbose=0)
print('train loss: {}, train accuracy: {},train precision: {},train recall: {},train fmeasure: {},train TP: {},train TN: {},train FP: {},train FN: {},'.format(train_score[0], train_score[1],train_score[2],train_score[3],train_score[4],train_score[5],train_score[6],train_score[7],train_score[8]))
test_score = model.evaluate(x_test, y_test, verbose=0)
print('test loss: {}, test accuracy: {},test precision: {},test recall: {},test fmeasure: {},test TP: {},test TN: {},test FP: {},test FN: {},'.format(test_score[0], test_score[1],test_score[2],test_score[3],test_score[4],test_score[5],test_score[6],test_score[7],test_score[8]))
print("---------------CNN-----------------")
input_shape = (img_x, img_y, img_z)

# convert the data to the right type
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, img_z)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, img_z)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(keras.layers.BatchNormalization())
print(model.output_shape)

shape=(model.output_shape)
#model.add(keras.layers.Reshape((32,-1)))
#model.add(SeqSelfAttention(attention_activation='sigmoid'))
print(model.output_shape)

#model.add((keras.layers.Reshape((29,29,32))))
#print(model.output_shape)

model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.BatchNormalization())
print(model.output_shape)

model.add(keras.layers.Reshape((64,-1)))
model.add(SeqSelfAttention(attention_activation='sigmoid'))
#model.add(SeqSelfAttention(
#    attention_width=15,
#    attention_activation='sigmoid',
#    name='Attention',
#))
model.add(keras.layers.Reshape((12,12,64)))
#model.add(keras.layers.Reshape((4,4,64)))

model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy',keras_metrics.precision(),keras_metrics.recall(), keras_metrics.f1_score(),keras_metrics.true_positive(),keras_metrics.true_negative(),keras_metrics.false_positive(),keras_metrics.false_negative()])


history3=model.fit(x_train, y_train, batch_size=batch_size, validation_split=0.2,
          epochs=epochs, verbose=1)
model.summary()
train_score = model.evaluate(x_train, y_train, verbose=0)
print('train loss: {}, train accuracy: {},train precision: {},train recall: {},train fmeasure: {},train TP: {},train TN: {},train FP: {},train FN: {},'.format(train_score[0], train_score[1],train_score[2],train_score[3],train_score[4],train_score[5],train_score[6],train_score[7],train_score[8]))
test_score = model.evaluate(x_test, y_test, verbose=0)
print('test loss: {}, test accuracy: {},test precision: {},test recall: {},test fmeasure: {},test TP: {},test TN: {},test FP: {},test FN: {},'.format(test_score[0], test_score[1],test_score[2],test_score[3],test_score[4],test_score[5],test_score[6],test_score[7],test_score[8]))

print("-----------------LSTM-------------")

x_train = x_train.reshape(x_train.shape[0],img_z,img_x, img_y)
x_test = x_test.reshape(x_test.shape[0],img_z, img_x, img_y)
x_train = x_train.reshape(x_train.shape[0], 4, -1)
x_test = x_test.reshape(x_test.shape[0], 4, -1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
model = Sequential()
model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(img_z, img_x*img_y)))
model.add(Bidirectional(LSTM(20, return_sequences=True)))
model.add(SeqSelfAttention(attention_activation='sigmoid'))

model.add(Bidirectional(LSTM(20, return_sequences=False)))

model.add((Dense(2, activation='sigmoid')))


model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy',keras_metrics.precision(),keras_metrics.recall(), keras_metrics.f1_score(),keras_metrics.true_positive(),keras_metrics.true_negative(),keras_metrics.false_positive(),keras_metrics.false_negative()])


history4=model.fit(x_train, y_train, batch_size=batch_size, validation_split=0.2,
          epochs=epochs, verbose=1)
model.summary()
train_score = model.evaluate(x_train, y_train, verbose=0)
print('train loss: {}, train accuracy: {},train precision: {},train recall: {},train fmeasure: {},train TP: {},train TN: {},train FP: {},train FN: {},'.format(train_score[0], train_score[1],train_score[2],train_score[3],train_score[4],train_score[5],train_score[6],train_score[7],train_score[8]))
test_score = model.evaluate(x_test, y_test, verbose=0)
ypred=np.round(model.predict(x_test))
cm = confusion_matrix(y_true=test_labels, y_pred=ypred)
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
print('test loss: {}, test accuracy: {},test precision: {},test recall: {},test fmeasure: {},test TP: {},test TN: {},test FP: {},test FN: {},'.format(test_score[0], test_score[1],test_score[2],test_score[3],test_score[4],test_score[5],test_score[6],test_score[7],test_score[8]))
plt.style.use('seaborn-darkgrid')

plt.plot(history1.history['accuracy'],'-o',markersize=4,linewidth=2)
plt.plot(history1.history['val_accuracy'],'d-',markersize=4,linewidth=2)
plt.plot(history2.history['accuracy'],'x-',markersize=4,linewidth=2)
plt.plot(history2.history['val_accuracy'],'*-',markersize=4,linewidth=2)
plt.plot(history3.history['accuracy'],'>-',markersize=4,linewidth=2)
plt.plot(history3.history['val_accuracy'],'p-',markersize=4,linewidth=2)
plt.plot(history4.history['accuracy'],'s-',markersize=4,linewidth=2)
plt.plot(history4.history['val_accuracy'],'p-',markersize=4,linewidth=2)
plt.plot([statistics.mean(k) for k in zip(history1.history['accuracy'], history2.history['accuracy'],history3.history['accuracy'],history4.history['accuracy'])],'X-',markersize=6,linewidth=3)
plt.plot([statistics.mean(k) for k in zip(history1.history['val_accuracy'], history2.history['val_accuracy'],history3.history['val_accuracy'],history4.history['val_accuracy'])],'P-',markersize=6,linewidth=3)

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
legend_properties = {'weight':'bold'}
plt.legend(['Att-MC-CLSTM accuracy', 'Att-MC-CLSTM val_accuracy','Att-MC-CNN-LSTM accuracy','Att-MC-CNN-LSTM val_accuracy','Att-MC-CNN accuracy','Att-MC-CNN val_accuracy','Att-MC-BiLSTM accuracy','Att-MC-BiLSTM val_accuracy','Average accuracy','Average val_accuracy'], loc='lower right',prop=legend_properties)

plt.show()

plt.style.use('seaborn-darkgrid')

plt.plot(history1.history['precision'],'-o',markersize=4,linewidth=2)
plt.plot(history1.history['val_precision'],'d-',markersize=4,linewidth=2)
plt.plot(history2.history['precision'],'x-',markersize=4,linewidth=2)
plt.plot(history2.history['val_precision'],'*-',markersize=4,linewidth=2)
plt.plot(history3.history['precision'],'>-',markersize=4,linewidth=2)
plt.plot(history3.history['val_precision'],'p-',markersize=4,linewidth=2)
plt.plot(history4.history['precision'],'s-',markersize=4,linewidth=2)
plt.plot(history4.history['val_precision'],'p-',markersize=4,linewidth=2)
plt.plot([statistics.mean(k) for k in zip(history1.history['accuracy'], history2.history['accuracy'],history3.history['accuracy'],history4.history['accuracy'])],'X-',markersize=6,linewidth=3)
plt.plot([statistics.mean(k) for k in zip(history1.history['val_accuracy'], history2.history['val_accuracy'],history3.history['val_accuracy'],history4.history['val_accuracy'])],'P-',markersize=6,linewidth=3)

plt.title('model precision')
plt.ylabel('precision')
plt.xlabel('epoch')
legend_properties = {'weight':'bold'}
plt.legend(['Att-MC-CLSTM precision', 'Att-MC-CLSTM val_precision','Att-MC-CNN-LSTM precision','Att-MC-CNN-LSTM val_precision','Att-MC-CNN precision','Att-MC-CNN val_precision','Att-MC-BiLSTM precision','Att-MC-BiLSTM val_precision','Average precision','Average val_precision'], loc='lower right',prop=legend_properties)

plt.show()
plt.style.use('seaborn-darkgrid')

plt.plot(history1.history['recall'],'-o',markersize=4,linewidth=2)
plt.plot(history1.history['val_recall'],'d-',markersize=4,linewidth=2)
plt.plot(history2.history['recall'],'x-',markersize=4,linewidth=2)
plt.plot(history2.history['val_recall'],'*-',markersize=4,linewidth=2)
plt.plot(history3.history['recall'],'>-',markersize=4,linewidth=2)
plt.plot(history3.history['val_recall'],'p-',markersize=4,linewidth=2)
plt.plot(history4.history['recall'],'s-',markersize=4,linewidth=2)
plt.plot(history4.history['val_recall'],'p-',markersize=4,linewidth=2)
plt.plot([statistics.mean(k) for k in zip(history1.history['accuracy'], history2.history['accuracy'],history3.history['accuracy'],history4.history['accuracy'])],'X-',markersize=6,linewidth=3)
plt.plot([statistics.mean(k) for k in zip(history1.history['val_accuracy'], history2.history['val_accuracy'],history3.history['val_accuracy'],history4.history['val_accuracy'])],'P-',markersize=6,linewidth=3)

plt.title('model recall')
plt.ylabel('recall')
plt.xlabel('epoch')
legend_properties = {'weight':'bold'}
plt.legend(['Att-MC-CLSTM recall', 'Att-MC-CLSTM val_recall','Att-MC-CNN-LSTM recall','Att-MC-CNN-LSTM val_recall','Att-MC-CNN recall','Att-MC-CNN val_recall','Att-MC-BiLSTM recall','Att-MC-BiLSTM val_recall','Average recall','Average val_recall'], loc='lower right',prop=legend_properties)

plt.show()
plt.style.use('seaborn-darkgrid')


plt.plot(history1.history['f1_score'],'-o',markersize=4,linewidth=2)
plt.plot(history1.history['val_f1_score'],'d-',markersize=4,linewidth=2)
plt.plot(history2.history['f1_score'],'x-',markersize=4,linewidth=2)
plt.plot(history2.history['val_f1_score'],'*-',markersize=4,linewidth=2)
plt.plot(history3.history['f1_score'],'>-',markersize=4,linewidth=2)
plt.plot(history3.history['val_f1_score'],'p-',markersize=4,linewidth=2)
plt.plot(history4.history['f1_score'],'s-',markersize=4,linewidth=2)
plt.plot(history4.history['val_f1_score'],'p-',markersize=4,linewidth=2)
plt.plot([statistics.mean(k) for k in zip(history1.history['accuracy'], history2.history['accuracy'],history3.history['accuracy'],history4.history['accuracy'])],'X-',markersize=6,linewidth=3)
plt.plot([statistics.mean(k) for k in zip(history1.history['val_accuracy'], history2.history['val_accuracy'],history3.history['val_accuracy'],history4.history['val_accuracy'])],'P-',markersize=6,linewidth=3)

plt.title('model f1_score')
plt.ylabel('f1_score')
plt.xlabel('epoch')
legend_properties = {'weight':'bold'}
plt.legend(['Att-MC-CLSTM f1_score', 'Att-MC-CLSTM val_f1_score','Att-MC-CNN-LSTM f1_score','Att-MC-CNN-LSTM val_f1_score','Att-MC-CNN f1_score','Att-MC-CNN val_f1_score','Att-MC-BiLSTM f1_score','Att-MC-BiLSTM val_f1_score','Average f1_score','Average val_f1_score'], loc='lower right',prop=legend_properties)

plt.show()