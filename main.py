import pandas as pd
import plotly.express as px
import streamlit as st
# import librosa
# import librosa.display
from scipy.io import wavfile
import time
import numpy as np
import pickle
import sounddevice as sd
import soundfile
from scipy.io.wavfile import write
from python_speech_features import mfcc

import keras
from keras.models import Sequential, model_from_json


if "audio" not in st.session_state:
    st.session_state.audio="not done"
    
if "upload" not in st.session_state:
    st.session_state.upload="not done"
    
if "predict" not in st.session_state:
    st.session_state.predict="not done"
    
def reset_state():
    st.session_state.audio="not done"
    st.session_state.upload="not done"
    st.session_state.predict="not done"


header = st.container()
demo = st.container()
headerCol1, headerCol2= st.columns(2)
col1, col2, col3 = st.columns([1,2,1])
feature = st.container()
    
def change_audio_state(filename):
    st.session_state["audio"]=filename

def change_audio_upload_state():
    st.session_state.upload="done"    

def recording():
    # CHUNK = 1024 
    # FORMAT = pyaudio.paInt16
    # CHANNELS = 2 
    # RATE = 44100
    # RECORD_SECONDS = 4
    WAVE_OUTPUT_FILENAME = "record.wav"

    # p = pyaudio.PyAudio()

    # stream = p.open(format=FORMAT,
    #                 channels=CHANNELS,
    #                 rate=RATE,
    #                 input=True,
    #                 frames_per_buffer=CHUNK) 


    # frames = []
    # progress_bar_status=st.progress(0)
    
    # _range = int(RATE / CHUNK * RECORD_SECONDS)

    # for i in range(0, _range):
    #     data = stream.read(CHUNK)
    #     frames.append(data)
    #     progress_bar_status.progress(i/_range)
    # st.success("Success record audio file")


    # stream.stop_stream()
    # stream.close()
    # p.terminate()

    # wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    # wf.setnchannels(CHANNELS)
    # wf.setsampwidth(p.get_sample_size(FORMAT))
    # wf.setframerate(RATE)
    # wf.writeframes(b''.join(frames))
    # wf.close()
    
    fs = 44100  # Sample rate
    seconds = 4  # Duration of recording
    
    

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    
    sd.wait()
    # progress_bar_status = st.progress(0) 
    # for i in range(0, seconds):
    #     time.sleep(0.01)
    #     progress_bar_status.progress(i/seconds)
 # Wait until recording is finished
    write(WAVE_OUTPUT_FILENAME, fs, myrecording) 
    
    # st.success("Success record audio file")
    change_audio_state(WAVE_OUTPUT_FILENAME)
    change_audio_upload_state()
    
def load_result_record():
    
    # data, sampling_rate = librosa.load(st.session_state.audio
    #                         ,res_type='kaiser_fast'
    #                         ,duration=2.5
    #                         ,sr=44100
    #                         ,offset=0.5)
    
    sampling_rate, data = wavfile.read(st.session_state.audio)
    
    df = pd.DataFrame(data=data, columns=['Amplitude', 'Amplitude2'])
    df.index = [(1/sampling_rate)*i for i in range(len(df.index))]
    
    fig =px.line(data_frame=df, y="Amplitude", labels="time")
    fig.update_layout(width=800)
    
    st.metric(label="Your file", value=st.session_state.audio, delta="sampling_rate : " + str(sampling_rate), delta_color="normal")
    st.write(fig)
    st.audio(data=st.session_state.audio, format="audio/wav")
    # st.write(data)
    
def loaded_model():
    json_file = open('model_json.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model
    
def action_predict():
    model = loaded_model()
    # X, sample_rate = librosa.load(st.session_state["audio"]
    #                           ,res_type='kaiser_fast'
    #                           ,duration=2.5
    #                           ,sr=44100
    #                           ,offset=0.5
    #                          )
    sampling_rate, X = wavfile.read(st.session_state.audio,mmap=False)
    sampling_rate = np.array(sampling_rate)
    mfccs = np.mean(mfcc(X, samplerate=sampling_rate, numcep=216, nfilt=216),axis=0)
    new_df = pd.DataFrame(data=mfccs).T
    new_df= np.expand_dims(new_df, axis=2)
    newpred = model.predict(new_df, 
                            batch_size=16, 
                            verbose=1)

    infile = open('labels','rb')
    lb = pickle.load(infile)
    infile.close()
    final = newpred.argmax(axis=1)
    final = final.astype(int).flatten()
    final = (lb.inverse_transform((final)))
    st.session_state.predict = final
    # st.write(final)
    
    
    
# ===================================
with header:
    st.markdown(" # Welcome to my project")
    st.markdown("Introduction")

with demo:
    st.markdown("There is a demo")
    col1.markdown("Record or Upload a file")

    upload_audio_file = headerCol1.file_uploader("Upload your audio file", accept_multiple_files=False, on_change=change_audio_upload_state)
    
    is_recording = headerCol2.button(on_click=recording, label="Click to Record")


    if st.session_state.upload == "done":
        progress_bar=col2.progress(0)

        for perc_completed in range(100):
            time.sleep(0.05)
            progress_bar.progress(perc_completed + 1)

        col2.success("Success get audio file")

        if is_recording == False:
            if upload_audio_file is not None:
                change_audio_state(upload_audio_file.name)
        
    
with feature:
    st.write(st.session_state["audio"])
    if st.session_state["audio"] != "not done":
        load_result_record()
        
    st.button(on_click=action_predict, label="Click to predict")
    if st.session_state.predict != "not done":
        st.metric(label="Your file", value=st.session_state.audio, delta="sampling_rate : 44100", delta_color="normal")
        st.write(st.session_state.predict)
        reset_state()
    
