import pandas as pd
import streamlit as st
import librosa
import librosa.display
from scipy.io import wavfile
from datetime import datetime  
import numpy as np
import pickle
from scipy.io.wavfile import write
from python_speech_features import mfcc
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tensorflow.keras.models import load_model, model_from_json, Sequential
from tensorflow.keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout,
                          GlobalMaxPool2D, MaxPool2D, concatenate, Activation, Input, Dense)
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="SER web-app", page_icon=":speech_balloon:", layout="wide")

model = load_model("saved_models/Emotion_Model_CNN2D.h5")

CAT8 = ["angry",
        "calm",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise"]
CAT3 = ["positive", "neutral", "negative"]
CAT16 = ['female_angry', 
         'female_calm', 
         'female_disgust', 
         'female_fear', 
         'female_happy', 
         'female_neutral', 
         'female_sad', 
         'female_surprise', 
         'male_angry', 
         'male_calm',  
         'male_disgust', 
         'male_fear', 
         'male_happy', 
         'male_neutral', 
         'male_sad', 
         'male_surprise']

TEST_CAT = ['fear', 'calm','disgust', 'neutral', 'happy', 'sad', 'surprise', 'angry']

COLOR_DICT_CAT3 = {"positive": "green",
                   "neutral": "blue",
                   "negative": "red"}

COLOR_DICT_CAT7 = {"angry": "red",
              "calm": "blue",
              "disgust": "cyan",
              "fear": "orange",
              "happy": "purple",
              "neutral": "black",
              "sad": "yellow",
              "surprise": "green"}

def getDateTimeNow():
    return datetime.now().strftime("%d-%m-%Y %H:%M:%S")

def log_file(txt, size):
    with open("log.txt", "a") as f:
        dateTimeNow = getDateTimeNow()
        f.write(f"{txt} - {size} - {dateTimeNow};\n")
        
def get_filename():
    dateTimeNow = getDateTimeNow()
    filename = dateTimeNow + "-record"
    return filename

def recording():
    return True
        
def save_audio(file):
    if file.size > 4000000:
        return 1
    
    folder = "audio"
    dateTimeNow = getDateTimeNow()
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try: 
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            
    try:
        with open("log.txt", "a") as f:
            f.write(f"{file.name} - {file.size} - {dateTimeNow}\n")
    except:
        pass
    
    with open(os.path.join(folder, file.name), "wb") as f:
        f.write(file.getbuffer())
    return 0

def get_melspec(audio):
    y, sampling_rate = librosa.load(audio, res_type='kaiser_fast'
                              ,duration=2.5
                              ,sr=44100
                              ,offset=0.5)
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    img = np.stack((Xdb,) * 3, -1)
    img = img.astype(np.uint8)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.resize(grayImage, (224, 224))
    rgbImage = np.repeat(grayImage[..., np.newaxis], 3, -1)
    return (rgbImage, Xdb)

def get_mfccs(audio):
    n = 30
    newdf = np.empty(shape=(1, n, 216, 1))
    X, sampling_rate = librosa.load(audio
                              ,res_type='kaiser_fast'
                              ,duration=2.5
                              ,sr=44100
                              ,offset=0.5
                             )
    sampling_rate = np.array(sampling_rate)
    mfccs = librosa.feature.mfcc(X, sr=sampling_rate, n_mfcc=n)
    mfccs = np.expand_dims(mfccs, axis=2)
    newdf[0,] = mfccs
    return newdf


@st.cache
def get_title(predictions, categories="001"):
    title = f"Detected emotion: {categories[predictions.argmax()]} - {predictions.max() * 100:.2f}%"
    return title

@st.cache
def color_dict(COLOR_DICT):
    return COLOR_DICT

## =====================================
def convert_data_list(predictions, categories):    
    dataList = np.array([])
    
    if len(categories) == 3:
        positive = predictions[0][4] + predictions[0][12] + predictions[0][7] + predictions[0][15]
        neutral = predictions[0][5] + predictions[0][13] + predictions[0][1] + predictions[0][9]
        negative = predictions[0][0] + predictions[0][8] + predictions[0][2] + predictions[0][10] + predictions[0][3] + predictions[0][11] + predictions[0][6] + predictions[0][14]
        dataList =np.array([positive, neutral, negative])
    if len(categories) == 8:
        angry = predictions[0][0] + predictions[0][8]
        calm = predictions[0][1] + predictions[0][9]
        disgust = predictions[0][2] + predictions[0][10]
        fear = predictions[0][3] + predictions[0][11]
        happy = predictions[0][4] + predictions[0][12]
        neutral = predictions[0][5] + predictions[0][13]
        sad = predictions[0][6] + predictions[0][14]
        surprise = predictions[0][7] + predictions[0][15]
        
        dataList =np.array([angry, calm, disgust, fear, happy, neutral, sad, surprise])
    if len(categories) == 16:
        return predictions
    return dataList

def convert_result(predictions, categories):
    ind = predictions.argmax()
    result = -1
    if len(categories) == 3:
        positive = [4, 12, 7, 15]
        neutral = [5, 13, 1, 9]
        negative = [0, 8, 2, 10, 3, 11, 6, 14]
        if ind in positive:
            result = 0
        if ind in neutral:
            result = 1 
        if ind in negative:
            result = 2  
    if len(categories) == 8:
        angry = [0, 8]
        calm = [1, 9]
        disgust = [2, 10]
        fear = [3, 11]
        happy = [4, 12]
        neutral = [5, 13]
        sad = [6, 14]
        surprise = [7, 15]
        if ind in angry:
            result = 0
        if ind in calm:
            result = 1
        if ind in disgust:
            result = 2
        if ind in fear:
            result = 3
        if ind in happy:
            result = 4
        if ind in neutral:
            result = 5
        if ind in sad:
            result = 6
        if ind in surprise:
            result = 7 
    if len(categories) == 16:
        return result
    return (result, len(categories))
        

st.cache(allow_output_mutation=True)    
def plot_colored_polar(fig, predictions, categories,
                        title="", colors=COLOR_DICT_CAT7):
    
    N = len(predictions)
    ind = convert_result(predictions=predictions, categories=categories)[0]
    # st.write(f"{ind} -- {N}")

    COLOR = colors[categories[ind]]
    sector_colors = [colors[i] for i in categories]

    fig.set_facecolor("#d1d1e0")
    ax = plt.subplot(111, polar="True")

    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    for sector in range(predictions.shape[0]):
        radii = np.zeros_like(predictions)
        radii[sector] = predictions[sector] * 10
        width = np.pi / 1.8 * predictions
        c = sector_colors[sector]
        ax.bar(theta, radii, width=width, bottom=0.0, color=c, alpha=0.25)

    angles = [i / float(N) * 2 * np.pi for i in range(N)]
    angles += angles[:1]

    data = list(predictions)
    data += data[:1]
    plt.polar(angles, data, color=COLOR, linewidth=2)
    plt.fill(angles, data, facecolor=COLOR, alpha=0.25)

    ax.spines['polar'].set_color('lightgrey')
    ax.set_theta_offset(np.pi / 3)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0, .25, .5, .75, 1], color="grey", size=8)

    plt.suptitle(title, color="darkblue", size=10)
    plt.title(f"BIG {N}\n", color=COLOR)
    plt.ylim(0, 1)
    plt.subplots_adjust(top=0.75)
    
st.cache(allow_output_mutation=True)
def predict_gender(predictions, categories):
    ind = predictions.argmax()
    pred = categories[ind]
    gender = ""
    if "male" in pred:
        gender = "image/user_male_icon.png"
    if "female" in pred:
        gender = "image/user_female_icon.png"
    return gender

## =====================================
def main(): 
    # side_image = Image.open("image/user_male_icon.png")
    # with st.sidebar:
    #     st.image(side_image, width=250)
    st.sidebar.header("Menu")
    website_menu = st.sidebar.selectbox("Menu", ("Emotion Recognition", "Dataset for training","About Me"))
    st.set_option('deprecation.showfileUploaderEncoding', False)
    
    if website_menu == "Emotion Recognition":
        st.sidebar.subheader("Model")
        model_type = st.sidebar.selectbox("Which type of present do you like?", ("mfccs", ""))
        em3 = gender = False
        st.sidebar.subheader("Setting")
        st.markdown("## Upload the file")
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                audio_file = st.file_uploader("Upload audio file", type=['wav', 'mp3', 'ogg'])
                if audio_file is not None:
                    if not os.path.exists("audio"):
                        os.makedirs("audio")
                    path = os.path.join("audio", audio_file.name)
                    is_saved_audio = save_audio(audio_file)
                    if is_saved_audio == 1:
                        st.warning("File size is too large. Try another file.")
                    elif is_saved_audio == 0:
                        st.audio(audio_file, format="audio/wav", start_time=0)
                        try:
                            wav, sampling_rate = librosa.load(path, res_type='kaiser_fast'
                              ,duration=2.5
                              ,sr=44100
                              ,offset=0.5)
                            Xdb = get_melspec(path)[1]
                            mfccs = librosa.feature.mfcc(wav, sr=sampling_rate)
                        except Exception as e:
                            audio_file = None
                            st.error(f"Error {e} - wrong format of the file. Try another .wav file.")
                    else:
                        st.error("Unknown error")
                else:
                    if st.button("Try test file"):
                        _testFile = "test.wav"
                        wav, sampling_rate = librosa.load(_testFile, res_type='kaiser_fast'
                              ,duration=2.5
                              ,sr=44100
                              ,offset=0.5)
                        Xdb = get_melspec(_testFile)[1]
                        mfccs = librosa.feature.mfcc(wav, sr=sampling_rate)
                        st.audio(_testFile, format='audio/wav', start_time=0)
                        path =  _testFile
                        audio_file = "test"
            with st.container():
                if audio_file is not None:
                    fig = plt.figure(figsize=(10,2))
                    fig.set_facecolor('#d1d1e0')
                    plt.title("Wave-form")
                    librosa.display.waveshow(wav, sr=44100)
                    plt.gca().axes.get_yaxis().set_visible(False)
                    plt.gca().axes.get_xaxis().set_visible(False)
                    plt.gca().axes.spines["right"].set_visible(False)
                    plt.gca().axes.spines["left"].set_visible(False)
                    plt.gca().axes.spines["top"].set_visible(False)
                    plt.gca().axes.spines["bottom"].set_visible(False)
                    plt.gca().axes.set_facecolor('#d1d1e0')
                    st.write(fig) 
                else:
                    pass      
            
            
        if model_type == "mfccs":
            em3 = st.sidebar.checkbox("3 emotions")
            em8 = st.sidebar.checkbox("8 emotions", True)
            gender = st.sidebar.checkbox("gender")

        elif model_type == "mel-specs":
            st.sidebar.warning("This model is temporarily disabled")

        else:
            st.sidebar.warning("This model is temporarily disabled")
            
        if audio_file is not None:
            st.markdown("## Analyzing...")
            if not audio_file == "test":
                st.sidebar.subheader("Your audio")
                st.sidebar.metric(label="Audio file", value=audio_file.name, delta=audio_file.size, delta_color="normal")
                log_file(audio_file.name, audio_file.size)
            if audio_file == "test":
                st.sidebar.subheader("Your audio")
                st.sidebar.metric(label="Try test file", value="test.wav")
            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    fig = plt.figure(figsize=(10, 2))
                    fig.set_facecolor('#d1d1e0')
                    plt.title("MFCCs")
                    librosa.display.specshow(mfccs, sr=sampling_rate, x_axis='time')
                    plt.gca().axes.get_yaxis().set_visible(False)
                    plt.gca().axes.spines["right"].set_visible(False)
                    plt.gca().axes.spines["left"].set_visible(False)
                    plt.gca().axes.spines["top"].set_visible(False)
                    st.write(fig)
                with col2:
                    fig2 = plt.figure(figsize=(10, 2))
                    fig2.set_facecolor('#d1d1e0')
                    plt.title("Mel-log-spectrogram")
                    librosa.display.specshow(Xdb, sr=sampling_rate, x_axis='time', y_axis='hz')
                    plt.gca().axes.get_yaxis().set_visible(False)
                    plt.gca().axes.spines["right"].set_visible(False)
                    plt.gca().axes.spines["left"].set_visible(False)
                    plt.gca().axes.spines["top"].set_visible(False)
                    st.write(fig2)
            if model_type == "mfccs":
                st.markdown("## Predictions")
                with st.container():
                    col1, col2, col3 = st.columns(3)
                    
                    model_ = model
                    mfccs_ = get_mfccs(path)
                    pred_ = model_.predict(mfccs_,batch_size=16)

                    with col1: 
                        if em3:
                            data3 = convert_data_list(pred_, CAT3)
                            txt = "MFCCs\n" + get_title(data3, CAT3)
                            fig = plt.figure(figsize=(5, 5))
                            COLORS = color_dict(COLOR_DICT_CAT3)
                            plot_colored_polar(fig, predictions=data3, categories=CAT3, title=txt, colors=COLORS)
                            st.write(fig)
                    with col2:
                        if em8:                    
                            data8 = convert_data_list(pred_, CAT8)
                            txt = "MFCCs\n" + get_title(pred_, CAT8)
                            fig3 = plt.figure(figsize=(5, 5))
                            COLORS = color_dict(COLOR_DICT_CAT7)
                            plot_colored_polar(fig3, predictions=data8, categories=CAT8, title=txt, colors=COLORS)
                            st.write(fig3)
                    with col3:
                        if gender:
                            with st.spinner('Wait for it...'):
                                gender = predict_gender(pred_, CAT16)
                                side_image = Image.open(gender)
                                st.image(side_image, width=300)
    elif website_menu == "Dataset for training":
        dataset = pd.read_csv("Data_path.csv")
        dataset.columns = ['Label', 'Source', 'Path/Filename']  
        st.header("My Training Dataset")
        st.write(dataset)  
        
        data_options = dataset['Path/Filename'].to_list()
        option = st.multiselect("Which file you want see?", data_options, ['data/RAVDESS/Actor_01/03-01-02-01-02-01-01.wav'])
        
        fig = plt.figure(figsize=(20, 5), facecolor='#d1d1e0') 
        _option = st.selectbox("Select number of MFCCs", [1,2,3,4,5,6,7,8,9,10,11,12,13], index=0)
        for opt in option:
            filename = opt
            opt = opt.split('/')[3]
            X, sampling_rate = librosa.load(filename
                              ,res_type='kaiser_fast'
                              ,sr=44100)
            opt_value = librosa.feature.mfcc(y=X, sr=sampling_rate, n_mfcc=30)
            for i in range(_option):
                plt.plot(opt_value[i], label=opt)
        
        plt.legend()
        plt.grid(axis='y', linestyle='-')
        st.subheader("Line-chart Comparing MFCCs of Audio")
        st.write(fig)
        
    elif website_menu == "About Me":
        st.subheader("About Me")
        st.balloons()
        col1, col2 = st.columns([3, 2])
        with col1:
            st.info("tamhltgcs18442@fpt.edu.vn")
            st.markdown(f""":speech_balloon: [Tam Huynh](https://www.linkedin.com/in/owenhuynh)""",
                        unsafe_allow_html=True)
        with col2:
            liimg = Image.open("image/user_male_icon.png")
            st.image(liimg)
    else:
        st.header("Thank you for coming to my website")
        
if __name__ == '__main__':
    main()