import keras_self_attention
import pywt
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import keras
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential,load_model
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
from sklearn.metrics import multilabel_confusion_matrix
import tensorflow as tf
from keras.callbacks import ModelCheckpoint,EarlyStopping
#from tensorflow.keras.callbacks import EarlyStopping
import itertools
from sklearn.metrics import confusion_matrix

import tfanalysis
import pandas as pd
import statistics
from sklearn.model_selection import train_test_split

import utility
import format_data
from sklearn import preprocessing
import keras_metrics
import transformdata
import combine2
from keras.layers import Bidirectional
from keras.layers import LSTM


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels
activities_description = {
    1: 'no_sabotage',
    2: 'sabotage'
}
cm_plot_labels = ['no_sabotage','sabotage']
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix Subject 1',
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
    plt.show()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

def get_data():
    df_1=pd.read_csv("/Users/mahima/Downloads/sub13_1.csv")
    df_2=pd.read_csv("/Users/mahima/Downloads/sub13_2.csv")

    df_11=pd.read_csv("/Users/mahima/Downloads/sub15_3.csv")
    df_12=pd.read_csv("/Users/mahima/Downloads/sub15_4.csv")

    df_1,label_1=transformdata.get_data(df_1,0)
    df_2,label_2=transformdata.get_data(df_2,0)
 
    df_11,label_11=transformdata.get_data(df_11,1)
    df_12,label_12=transformdata.get_data(df_12,1)
  

    scales = range(1,64)
    test_score=[]
    train_data_cwt_1,train_labels_1,test_data_cwt,test_labels=combine2.combine_data(df_1,label_1,df_2,label_2,df_11,label_11,df_12,label_12)#,df_5,label_5,df_6,label_6,df_9,label_9,df_10,label_10,df_13,label_13,df_14,label_14)#,df_3,label_3,df_4,label_4,df_5,label_5,df_6,label_6,df_7,label_7,df_8,label_8,df_9,label_9,df_10,label_10,df_11,label_11,df_12,label_12,df_13,label_13,df_14,label_14,df_15,label_15,df_16,label_16)
    print("*********************************************************")
    

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
    #[no_signals_valid, no_steps_valid, no_components_valid] = np.shape(valid_data_cwt)

    train_data_cwt,train_labels=randomize(train_data_cwt_1,train_labels_1)
    test_data_cwt,test_labels=randomize(test_data_cwt,test_labels)
   # valid_data_cwt,valid_labels=randomize(valid_data_cwt,valid_labels)
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



    x_train = train_data_cwt_new
    y_train = train_labels
    x_test = test_data_cwt_new
    y_test = test_labels
    #x_valid = valid_data_cwt_new
    #y_valid = valid_labels
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


    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices - this is for use in the
    # categorical_crossentropy loss below
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    input_shape = (img_x, img_y,img_z)

    # convert the data to the right type
    #x_train = x_train.reshape(x_train.shape[0],img_z, img_x, img_y, 1)
    #x_test = x_test.reshape(x_test.shape[0], img_z,img_x, img_y, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    #y_valid = keras.utils.to_categorical(y_valid, num_classes)
    return x_train,x_test,y_train,y_test

def cnn_lstm():
    print("****************CNN LSTM*******************")
    img_x = 63
    img_y = 63
    img_z = 4
    input_shape = (img_x, img_y, img_z)

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add((Conv2D(filters=32, kernel_size=3, activation='relu')))
    print(model.output_shape)

    model.add((Dropout(0.5)))
    model.add((MaxPooling2D(pool_size=2)))
    model.add((Flatten()))
    model.add(keras.layers.Reshape((32, -1)))

    print(model.output_shape)
    model.add(LSTM(100, return_sequences=True)
              )
    # model.add(LSTM(100,return_sequences=True)
    # )
    print(model.output_shape)
    print("Hiiiii", model.input_shape)

    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    print(f"{model.output_shape} is attention")

    model.add(keras.layers.Reshape((32, 100)))
    print(f"{model.output_shape} is reshape")
    model.add(Flatten())
    model.add(Dropout(0.5))
    print(f"{model.output_shape} is flatten")

    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])#, keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score(),
                           #keras_metrics.true_positive(), keras_metrics.true_negative(), keras_metrics.false_positive(),
                           #keras_metrics.false_negative()])

    return model

pat = 5 #this is the number of epochs with no improvment after which the training will stop
early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)

#define the model checkpoint callback -> this will keep on saving the model as a physical file
model_checkpoint = ModelCheckpoint('fas_mnist_sab1514.h5', verbose=1, save_best_only=True)

#define a function to fit the model
def fit_and_evaluate_cnnlstm(t_x, val_x, t_y, val_y, EPOCHS=20, BATCH_SIZE=16):
    model = None
    model = cnn_lstm()
    results = model.fit(t_x, t_y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping, model_checkpoint],
              verbose=1, validation_split=0.1)
    print("Val Score: ", model.evaluate(val_x, val_y))
    return results


#save the model history in a list after fitting so that we can plot later
x_train, x_test, y_train, y_test=get_data()

n_folds = 5
epochs = 20
BATCH_SIZE = 16
model_history_1 = []
for i in range(n_folds):
    print("Training on Fold: ",i+1)
    t_x, val_x, t_y, val_y = train_test_split(x_train, y_train, test_size=0.1,
                                               random_state = np.random.randint(1,1000, 1)[0])
    model_history_1.append(fit_and_evaluate_cnnlstm(t_x, val_x, t_y, val_y, epochs, 16))
    print("======="*12, end="\n\n\n")
model = load_model('fas_mnist_sab1514.h5',custom_objects={'SeqSelfAttention': keras_self_attention.SeqSelfAttention})
train_score = model.evaluate(x_train, y_train, verbose=0)
#print('train loss: {}, train accuracy: {},train precision: {},train recall: {},train fmeasure: {},train TP: {},train TN: {},train FP: {},train FN: {},'.format(train_score[0], train_score[1],train_score[2],train_score[3],train_score[4],train_score[5],train_score[6],train_score[7],train_score[8]))
print('train loss: {}, train accuracy: {}'.format(train_score[0], train_score[1]))

test_score = model.evaluate(x_test, y_test, verbose=0)
#print('test loss: {}, test accuracy: {},test precision: {},test recall: {},test fmeasure: {},test TP: {},test TN: {},test FP: {},test FN: {},'.format(test_score[0], test_score[1],test_score[2],test_score[3],test_score[4],test_score[5],test_score[6],test_score[7],test_score[8]))
print('test loss: {}, train accuracy: {}'.format(test_score[0], test_score[1]))

ypred=np.round(model.predict(x_test))
cm = confusion_matrix(y_true=tf.argmax(y_test,1), y_pred=tf.argmax(ypred,1))
y_true=tf.argmax(y_test,1)
y_pred=tf.argmax(ypred,1)
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion matrix Subject 11')

plt.title('Att-MC-CNN-LSTM Train Accuracy vs Val Accuracy')
plt.style.use('seaborn-darkgrid')

plt.plot(model_history_1[0].history['accuracy'], label='Train Accuracy Fold 1', color='black')
plt.plot(model_history_1[0].history['val_accuracy'], label='Val Accuracy Fold 1', color='black', linestyle = "dashdot")
plt.plot(model_history_1[1].history['accuracy'], label='Train Accuracy Fold 2', color='red', )
plt.plot(model_history_1[1].history['val_accuracy'], label='Val Accuracy Fold 2', color='red', linestyle = "dashdot")
plt.plot(model_history_1[2].history['accuracy'], label='Train Accuracy Fold 3', color='green', )
plt.plot(model_history_1[2].history['val_accuracy'], label='Val Accuracy Fold 3', color='green', linestyle = "dashdot")
plt.plot(model_history_1[3].history['accuracy'], label='Train Accuracy Fold 4', color='purple', )
plt.plot(model_history_1[3].history['val_accuracy'], label='Val Accuracy Fold 4', color='purple', linestyle = "dashdot")
plt.plot(model_history_1[4].history['accuracy'], label='Train Accuracy Fold 5', color='blue', )
plt.plot(model_history_1[4].history['val_accuracy'], label='Val Accuracy Fold 5', color='blue', linestyle = "dashdot")
plt.legend()
plt.show()

def conv_lstm():
    print("************************CONVLSTM**************************")
    img_x = 63
    img_y = 63
    img_z = 4
    input_shape = (1, img_x, img_y, img_z)
    # convert the data to the right type

    model = Sequential()
    model.add(ConvLSTM2D(filters=32, kernel_size=(2, 2), strides=(1, 1), activation='relu', input_shape=(input_shape),
                         return_sequences=True))
    #model.add(keras.layers.BatchNormalization())
    model.add(ConvLSTM2D(filters=32, kernel_size=(2, 2), strides=(1, 1), activation='relu', input_shape=(input_shape),
                         return_sequences=True))
    #model.add(keras.layers.BatchNormalization())
    print(model.output_shape)
    model.add(keras.layers.Reshape((32, -1)))
    print(model.output_shape)
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    print(model.output_shape)
    model.add(keras.layers.Reshape((61, 61, 32)))
    # model.add(keras.layers.Reshape((30,30,32)))
    print(model.output_shape)
    model.add(Dropout(0.5))
    model.add(Flatten())
    print(model.output_shape)
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))


    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])#, keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score(),
                           #keras_metrics.true_positive(), keras_metrics.true_negative(), keras_metrics.false_positive(),
                           #keras_metrics.false_negative()])

    return model
pat = 5 #this is the number of epochs with no improvment after which the training will stop
early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)

#define the model checkpoint callback -> this will keep on saving the model as a physical file
model_checkpoint = ModelCheckpoint('fas_mnist_2.h5', verbose=1, save_best_only=True)

#define a function to fit the model
def fit_and_evaluate_convlstm(t_x, val_x, t_y, val_y, EPOCHS=20, BATCH_SIZE=16):
    model = None
    img_x = 63
    img_y = 63
    img_z = 4

    t_x = t_x.reshape(t_x.shape[0], 1, img_x, img_y, img_z)
    val_x = val_x.reshape(val_x.shape[0], 1, img_x, img_y, img_z)
    x_train = t_x.astype('float32')
    x_test = val_x.astype('float32')
    model = cnn_lstm()
    results = model.fit(t_x, t_y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping, model_checkpoint],
              verbose=1, validation_split=0.1)
    print("Val Score: ", model.evaluate(val_x, val_y))
    return results

n_folds=5
epochs=20
BATCH_SIZE=16

#save the model history in a list after fitting so that we can plot later
model_history_2 = []
x_train, x_test, y_train, y_test=get_data()

for i in range(n_folds):
    print("Training on Fold: ",i+1)
    t_x, val_x, t_y, val_y = train_test_split(x_train, y_train, test_size=0.1,
                                               random_state = np.random.randint(1,1000, 1)[0])
    model_history_2.append(fit_and_evaluate_convlstm(t_x, val_x, t_y, val_y, epochs, 16))
    print("======="*12, end="\n\n\n")

model = load_model('fas_mnist_2.h5',custom_objects={'SeqSelfAttention': keras_self_attention.SeqSelfAttention})
img_x = 63
img_y = 63
img_z = 4
x_train = x_train.reshape(x_train.shape[0], 1,img_x, img_y, img_z)

x_test = x_test.reshape(x_test.shape[0],1, img_x, img_y, img_z)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
train_score = model.evaluate(x_train, y_train, verbose=0)
#print('train loss: {}, train accuracy: {},train precision: {},train recall: {},train fmeasure: {},train TP: {},train TN: {},train FP: {},train FN: {},'.format(train_score[0], train_score[1],train_score[2],train_score[3],train_score[4],train_score[5],train_score[6],train_score[7],train_score[8]))
print('train loss: {}, train accuracy: {}'.format(train_score[0], train_score[1]))

test_score = model.evaluate(x_test, y_test, verbose=0)
ypred=np.round(model.predict(x_test))
cm = confusion_matrix(y_true=y_test, y_pred=ypred)
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
#print('test loss: {}, test accuracy: {},test precision: {},test recall: {},test fmeasure: {},test TP: {},test TN: {},test FP: {},test FN: {},'.format(test_score[0], test_score[1],test_score[2],test_score[3],test_score[4],test_score[5],test_score[6],test_score[7],test_score[8]))
print('test loss: {}, train accuracy: {}'.format(test_score[0], test_score[1]))
plt.title('Att-MC-CLSTM Train Accuracy vs Val Accuracy')
plt.style.use('seaborn-darkgrid')

plt.xlim(0, 20)
plt.ylim(0, 1)
plt.plot(model_history_2[0].history['accuracy'], label='Train Accuracy Fold 1', color='black')
plt.plot(model_history_2[0].history['val_accuracy'], label='Val Accuracy Fold 1', color='black', linestyle = "dashdot")
plt.plot(model_history_2[1].history['accuracy'], label='Train Accuracy Fold 2', color='red', )
plt.plot(model_history_2[1].history['val_accuracy'], label='Val Accuracy Fold 2', color='red', linestyle = "dashdot")
plt.plot(model_history_2[2].history['accuracy'], label='Train Accuracy Fold 3', color='green', )
plt.plot(model_history_2[2].history['val_accuracy'], label='Val Accuracy Fold 3', color='green', linestyle = "dashdot")
plt.plot(model_history_2[3].history['accuracy'], label='Train Accuracy Fold 4', color='purple', )
plt.plot(model_history_2[3].history['val_accuracy'], label='Val Accuracy Fold 4', color='purple', linestyle = "dashdot")
plt.plot(model_history_2[4].history['accuracy'], label='Train Accuracy Fold 5', color='blue', )
plt.plot(model_history_2[4].history['val_accuracy'], label='Val Accuracy Fold 5', color='blue', linestyle = "dashdot")
plt.legend()
plt.show()

print('meant_1 =',model_history_2[0].history['accuracy'])
print('meanv_1 =',model_history_2[0].history['val_accuracy'])
print('meant_2 =',model_history_2[1].history['accuracy'])
print('meanv_2 =',model_history_2[1].history['val_accuracy'])
print('meant_3 =',model_history_2[2].history['accuracy'])
print('meanv_3 =',model_history_2[2].history['val_accuracy'])
print('meant_4 =',model_history_2[3].history['accuracy'])
print('meanv_4 =',model_history_2[3].history['val_accuracy'])
print('meant_5 =',model_history_2[4].history['accuracy'])
print('meanv_5 =',model_history_2[4].history['val_accuracy'])


def CNN():
    print("************************CNN**************************")

    img_x = 63
    img_y = 63
    img_z = 4
    input_shape = (img_x, img_y, img_z)

    # convert the data to the right type

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    print(model.output_shape)

    shape = (model.output_shape)
    # model.add(keras.layers.Reshape((32,-1)))
    # model.add(SeqSelfAttention(attention_activation='sigmoid'))
    print(model.output_shape)

    # model.add((keras.layers.Reshape((29,29,32))))
    # print(model.output_shape)

    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    print(model.output_shape)

    model.add(keras.layers.Reshape((64, -1)))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    # model.add(SeqSelfAttention(
    #    attention_width=15,
    #    attention_activation='sigmoid',
    #    name='Attention',
    # ))
    model.add(keras.layers.Reshape((12, 12, 64)))
    # model.add(keras.layers.Reshape((4,4,64)))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])#, keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score(),
                           #keras_metrics.true_positive(), keras_metrics.true_negative(), keras_metrics.false_positive(),
                           #keras_metrics.false_negative()])

    return model
pat = 5  # this is the number of epochs with no improvment after which the training will stop
early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)

# define the model checkpoint callback -> this will keep on saving the model as a physical file
model_checkpoint = ModelCheckpoint('fas_mnist_3.h5', verbose=1, save_best_only=True)


# define a function to fit the model
def fit_and_evaluate_CNN(t_x, val_x, t_y, val_y, EPOCHS=20, BATCH_SIZE=16):
    model = None
    model = CNN()

    img_x = 63
    img_y = 63
    img_z = 4
    t_x = t_x.reshape(t_x.shape[0], img_x, img_y, img_z)
    val_x = val_x.reshape(val_x.shape[0], img_x, img_y, img_z)
    x_train = t_x.astype('float32')
    x_test = val_x.astype('float32')
    results = model.fit(t_x, t_y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping, model_checkpoint],
                        verbose=1, validation_split=0.1)
    print("Val Score: ", model.evaluate(val_x, val_y))
    return results

n_folds = 5
epochs = 20
batch_size = 128

# save the model history in a list after fitting so that we can plot later
model_history_3 = []

for i in range(n_folds):
    print("Training on Fold: ", i + 1)
    t_x, val_x, t_y, val_y = train_test_split(x_train, y_train, test_size=0.1,
                                              random_state=np.random.randint(1, 1000, 1)[0])
    model_history_3.append(fit_and_evaluate_CNN(t_x, val_x, t_y, val_y, epochs, 16))
    print("=======" * 12, end="\n\n\n")

model = load_model('fas_mnist_3.h5',custom_objects={'SeqSelfAttention': keras_self_attention.SeqSelfAttention})

img_x = 63
img_y = 63
img_z = 4
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, img_z)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, img_z)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
train_score = model.evaluate(x_train, y_train, verbose=0)
#print('train loss: {}, train accuracy: {},train precision: {},train recall: {},train fmeasure: {},train TP: {},train TN: {},train FP: {},train FN: {},'.format(train_score[0], train_score[1],train_score[2],train_score[3],train_score[4],train_score[5],train_score[6],train_score[7],train_score[8]))
print('train loss: {}, train accuracy: {}'.format(train_score[0], train_score[1]))

test_score = model.evaluate(x_test, y_test, verbose=0)
#print('test loss: {}, test accuracy: {},test precision: {},test recall: {},test fmeasure: {},test TP: {},test TN: {},test FP: {},test FN: {},'.format(test_score[0], test_score[1],test_score[2],test_score[3],test_score[4],test_score[5],test_score[6],test_score[7],test_score[8]))
print('test loss: {}, train accuracy: {}'.format(test_score[0], test_score[1]))
plt.title('Att-MC-CNN Train Accuracy vs Val Accuracy')
plt.style.use('seaborn-darkgrid')

plt.plot(model_history_3[0].history['accuracy'], label='Train Accuracy Fold 1', color='black')
plt.plot(model_history_3[0].history['val_accuracy'], label='Val Accuracy Fold 1', color='black', linestyle = "dashdot")
plt.plot(model_history_3[1].history['accuracy'], label='Train Accuracy Fold 2', color='red', )
plt.plot(model_history_3[1].history['val_accuracy'], label='Val Accuracy Fold 2', color='red', linestyle = "dashdot")
plt.plot(model_history_3[2].history['accuracy'], label='Train Accuracy Fold 3', color='green', )
plt.plot(model_history_3[2].history['val_accuracy'], label='Val Accuracy Fold 3', color='green', linestyle = "dashdot")
plt.plot(model_history_3[3].history['accuracy'], label='Train Accuracy Fold 4', color='purple', )
plt.plot(model_history_3[3].history['val_accuracy'], label='Val Accuracy Fold 4', color='purple', linestyle = "dashdot")
plt.plot(model_history_3[4].history['accuracy'], label='Train Accuracy Fold 5', color='blue', )
plt.plot(model_history_3[4].history['val_accuracy'], label='Val Accuracy Fold 5', color='blue', linestyle = "dashdot")
plt.legend()
plt.show()

def LSTM_layer():
    print("************************LSTM**************************")
    model = Sequential()

    img_x = 63
    img_y = 63
    img_z = 4
    model.add(Bidirectional(LSTM(20,return_sequences=True), input_shape=(img_z, img_x * img_y)))
    model.add(Bidirectional(LSTM(20, return_sequences=True)))
    model.add(SeqSelfAttention(attention_activation='sigmoid'))

    model.add(Bidirectional(LSTM(20, return_sequences=False)))

    model.add((Dense(2, activation='sigmoid')))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])#, keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score(),
                           #keras_metrics.true_positive(), keras_metrics.true_negative(), keras_metrics.false_positive(),
                           #keras_metrics.false_negative()])

    return model

pat = 5  # this is the number of epochs with no improvment after which the training will stop
early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)

# define the model checkpoint callback -> this will keep on saving the model as a physical file
model_checkpoint = ModelCheckpoint('fas_mnist_4.h5', verbose=1, save_best_only=True)


# define a function to fit the model
def fit_and_evaluate_LSTM(t_x, val_x, t_y, val_y, EPOCHS=20, BATCH_SIZE=16):
    model = None
    model = LSTM_layer()


    img_x = 63
    img_y = 63
    img_z = 4
    t_x = t_x.reshape(t_x.shape[0], img_z, img_x, img_y)
    val_x = val_x.reshape(val_x.shape[0], img_z, img_x, img_y)
    t_x = t_x.reshape(t_x.shape[0], 4, -1)
    val_x = val_x.reshape(val_x.shape[0], 4, -1)
    results = model.fit(t_x, t_y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping, model_checkpoint],
                        verbose=1, validation_split=0.1)
    print("Val Score: ", model.evaluate(val_x, val_y))
    return results

x_train, x_test, y_train, y_test=get_data()

n_folds = 5
epochs = 20
batch_size = 128

# save the model history in a list after fitting so that we can plot later
model_history_4 = []

for i in range(n_folds):
    print("Training on Fold: ", i + 1)
    t_x, val_x, t_y, val_y = train_test_split(x_train, y_train, test_size=0.1,
                                              random_state=np.random.randint(1, 1000, 1)[0])
    model_history_4.append(fit_and_evaluate_LSTM(t_x, val_x, t_y, val_y, epochs, 16))
    print("=======" * 12, end="\n\n\n")

model = load_model('fas_mnist_4.h5',custom_objects={'SeqSelfAttention': keras_self_attention.SeqSelfAttention})


img_x = 63
img_y = 63
img_z = 4
x_train = x_train.reshape(x_train.shape[0],img_z,img_x, img_y)
x_test = x_test.reshape(x_test.shape[0],img_z, img_x, img_y)
x_train = x_train.reshape(x_train.shape[0], 4, -1)
x_test = x_test.reshape(x_test.shape[0], 4, -1)
train_score = model.evaluate(x_train, y_train, verbose=0)
#print('train loss: {}, train accuracy: {},train precision: {},train recall: {},train fmeasure: {},train TP: {},train TN: {},train FP: {},train FN: {},'.format(train_score[0], train_score[1],train_score[2],train_score[3],train_score[4],train_score[5],train_score[6],train_score[7],train_score[8]))
print('train loss: {}, train accuracy: {}'.format(train_score[0], train_score[1]))

test_score = model.evaluate(x_test, y_test, verbose=0)
#print('test loss: {}, test accuracy: {},test precision: {},test recall: {},test fmeasure: {},test TP: {},test TN: {},test FP: {},test FN: {},'.format(test_score[0], test_score[1],test_score[2],test_score[3],test_score[4],test_score[5],test_score[6],test_score[7],test_score[8]))
print('test loss: {}, train accuracy: {}'.format(test_score[0], test_score[1]))
plt.title('Att-MC-BiLSTM Train Accuracy vs Val Accuracy')
plt.style.use('seaborn-darkgrid')

plt.plot(model_history_4[0].history['accuracy'], label='Train Accuracy Fold 1', color='black')
plt.plot(model_history_4[0].history['val_accuracy'], label='Val Accuracy Fold 1', color='black', linestyle = "dashdot")
plt.plot(model_history_4[1].history['accuracy'], label='Train Accuracy Fold 2', color='red', )
plt.plot(model_history_4[1].history['val_accuracy'], label='Val Accuracy Fold 2', color='red', linestyle = "dashdot")
plt.plot(model_history_4[2].history['accuracy'], label='Train Accuracy Fold 3', color='green', )
plt.plot(model_history_4[2].history['val_accuracy'], label='Val Accuracy Fold 3', color='green', linestyle = "dashdot")
plt.plot(model_history_4[3].history['accuracy'], label='Train Accuracy Fold 4', color='purple', )
plt.plot(model_history_4[3].history['val_accuracy'], label='Val Accuracy Fold 4', color='purple', linestyle = "dashdot")
plt.plot(model_history_4[4].history['accuracy'], label='Train Accuracy Fold 5', color='blue', )
plt.plot(model_history_4[4].history['val_accuracy'], label='Val Accuracy Fold 5', color='blue', linestyle = "dashdot")
plt.legend()
plt.show()
