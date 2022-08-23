from imp import load_module
import streamlit as st

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import keras
from keras import regularizers
from keras.preprocessing import sequence
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
from tensorflow.keras import optimizers

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wave
import IPython.display as ipd 
import os
import pickle
import sys
import warnings
import pyaudio
import h5py


if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 


header = st.container()
dataset = st.container()
feature = st.container()
model_training = st.container()

# Source 
RAVDESS = "data/RAVDESS/"

@st.cache
def get_data(filename):
    taxi_data = pd.read_csv(filename)
    return taxi_data

@st.cache
def load_data():
    data = pd.read_csv('Data_path.csv')
    data_frame = pd.DataFrame(columns=['feature'])

    # loop feature extraction over the entire dataset
    counter=0
    for index,path in enumerate(data.path):
        X, sample_rate = librosa.load(path
                                    , res_type='kaiser_fast'
                                    ,duration=2.5
                                    ,sr=44100
                                    ,offset=0.5
                                    )
        sample_rate = np.array(sample_rate)
        
        # mean as the feature. Could do min and max etc as well. 
        mfccs = np.mean(librosa.feature.mfcc(y=X, 
                                            sr=sample_rate, 
                                            n_mfcc=13),
                        axis=0)
        data_frame.loc[counter] = [mfccs]
        counter=counter+1   

    # Check a few records to make sure its processed successfully

    data_frame = pd.concat([data,pd.DataFrame(data_frame['feature'].values.tolist())],axis=1)

    data_frame = data_frame.fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(data_frame.drop(['path','labels','source'],axis=1)
                                                    , data_frame.labels
                                                    , test_size=0.25
                                                    , shuffle=True
                                                    , random_state=42
                                                   )
    
    return [X_train, X_test, y_train, y_test]

def data_normalization(data):
    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]
        
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean)/std
    X_test = (X_test - mean)/std
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
        
    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))

    st.write(X_train.shape)
    st.write(lb.classes_)

    filename = 'labels'
    outfile = open(filename,'wb')
    pickle.dump(lb,outfile)
    outfile.close()
        
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
        
    st.write(X_train.shape)
    return [X_train, X_test, y_train, y_test, lb]

def creating_model(data):
    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]
    
    model = Sequential()
    model.add(Conv1D(256, 8, padding='same',input_shape=(X_train.shape[1],1)))
    model.add(Activation('relu'))
    model.add(Conv1D(256, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 8, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(64, 8, padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(14))
    model.add(Activation('softmax'))
    return model

@st.cache        
def training():
    data = load_data()
    st.dataframe(data[0])
        
    data = data_normalization(data) 
        
    model = creating_model(data)
    
    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]
    
    opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)  
    model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
    model_history=model.fit(X_train, y_train, batch_size=16, epochs=100, validation_data=(X_test, y_test))
    loss = model_history.history['loss']
    epoch = model_history.history['val_loss']
    figTrain = plt.figure(figsize=(20, 15))
    axTrain = plt.subplot(3,1,1)
    axTrain.plot(loss)
    axTrain.plot(epoch)
    # axTrain.title('model loss')
    # figTrain.ylabel('loss')
    # figTrain.xlabel('epoch')
    axTrain.legend(['train', 'test'], loc='upper left')
    st.pyplot(figTrain)
    
            
    model_name = 'Emotion_Model.h5'
    save_dir = os.path.join(os.getcwd(), 'saved_models')

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)

    model_json = model.to_json()
    with open('model_json.json', "w") as json_file:
        json_file.write(model_json)

    comparingTrainingResult(data)
    return 1

    
def get_file_audio(filename): 
    path = RAVDESS + filename
    X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=6,sr=22050*2,offset=0.5)  
    result = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
    result = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    return result

def loaded_model():
    json_file = open('model_json.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model

def predicting(filename, loaded_model):
    X, sample_rate = librosa.load(filename
                              ,res_type='kaiser_fast'
                              ,duration=2.5
                              ,sr=44100
                              ,offset=0.5
                             )
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
    new_df = pd.DataFrame(data=mfccs).T
    new_df= np.expand_dims(new_df, axis=2)
    newpred = loaded_model.predict(new_df, 
                            batch_size=16, 
                            verbose=1)

    infile = open('labels','rb')
    lb = pickle.load(infile)
    infile.close()
    final = newpred.argmax(axis=1)
    final = final.astype(int).flatten()
    final = (lb.inverse_transform((final)))
    st.write(final)
    
def comparingTrainingResult(data):
    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]
    lb = data[4]
    
    json_file = open('model_json.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("saved_models/Emotion_Model.h5")
    print("Loaded model from disk")
    
    # Keras optimiser
    opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
    loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    score = loaded_model.evaluate(X_test, y_test, verbose=0)
    st.write("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    preds = loaded_model.predict(X_test, 
                         batch_size=14, 
                         verbose=1)

    preds=preds.argmax(axis=1)
    
    # predictions 
    preds = preds.astype(int).flatten()
    preds = (lb.inverse_transform((preds)))
    preds = pd.DataFrame({'predictedvalues': preds})

    # Actual labels
    actual=y_test.argmax(axis=1)
    actual = actual.astype(int).flatten()
    actual = (lb.inverse_transform((actual)))
    actual = pd.DataFrame({'actualvalues': actual})

    # Lets combined both of them into a single dataframe
    finaldf = actual.join(preds)
    st.dataframe(finaldf)
    
    # Get the predictions file 
    finaldf = pd.read_csv("Predictions.csv")
    classes = finaldf.actualvalues.unique()
    classes.sort()    

    # Confusion matrix 
    c = confusion_matrix(finaldf.actualvalues, finaldf.predictedvalues)
    print(accuracy_score(finaldf.actualvalues, finaldf.predictedvalues))
    print_confusion_matrix(c, class_names = classes)

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
        
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    st.pyplot(fig)

def gender(row):
    if row == 'female_disgust' or 'female_fear' or 'female_happy' or 'female_sad' or 'female_surprise' or 'female_neutral':
        return 'female'
    elif row == 'male_angry' or 'male_fear' or 'male_happy' or 'male_sad' or 'male_surprise' or 'male_neutral' or 'male_disgust':
        return 'male'
    

def recording():
    CHUNK = 1024 
    FORMAT = pyaudio.paInt16
    CHANNELS = 2 
    RATE = 44100
    RECORD_SECONDS = 4
    WAVE_OUTPUT_FILENAME = "record.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK) #buffer

    st.write("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    st.write("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return WAVE_OUTPUT_FILENAME

# --------------- UI ----------------------


with header:
    st.title("WELCOME TO MY PROJECT")
    st.subheader('Introduction')
    st.text('This is a demo, you can click button below to start')
    isClicked = st.button('Start Record')
    if (isClicked):
        st.text('Say something to record you void')
        audio_record = recording()
        predicting(audio_record,loaded_model())
        data, sampling_rate = librosa.load(audio_record)
        st.audio(open(audio_record,'rb').read(),  format='audio/wav')
        fig = plt.figure(figsize=(15, 5))
        axis = plt.plot(np.arange(len(data))/sampling_rate, data)
        st.pyplot(fig)
    



with dataset:
    st.subheader('My Dataset')
    data = pd.read_csv('Data_path.csv', usecols=[0, 1])
    st.dataframe(data, width=1000)
    
    female = get_file_audio("Actor_08/03-01-05-02-01-01-08.wav")
    male = get_file_audio("Actor_09/03-01-05-01-01-01-09.wav")
    fig = plt.figure(figsize=(20, 15))
    ax = plt.subplot(3,1,1)
    ax.plot(female, label='female')
    ax.plot(male, label='male')
    fig.legend()
    st.pyplot(fig)




with model_training:
    st.header('model_training')
    st.text('model_training text')
    isReTraining = st.button('Re-training')
    if (isReTraining):
        training()
        
    st.subheader('')
        
    st.text('Using the new training result:')
    isClicked = st.button('Record')
    if (isClicked):
        st.text('Say something to record you void')
        audio_record = recording()
        predicting(audio_record,loaded_model())
        data, sampling_rate = librosa.load(audio_record)
        st.audio(open(audio_record,'rb').read(),  format='audio/wav')
        fig = plt.figure(figsize=(15, 5))
        axis = plt.plot(np.arange(len(data))/sampling_rate, data)
        st.pyplot(fig)

        