import streamlit as st
import pickle
import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import pyaudio
import wave

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".\models"))


with open(os.path.join(MODEL_DIR,"labelencoder.pkl"),"rb") as f:
    labelencoder = pickle.load(f)

st.set_page_config(page_title="Voice Classification System",page_icon="",)
st.title("Real time Voice Classification System")
max_pad_len=100
def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, res_type='kaiser_fast')
    y, _ = librosa.effects.trim(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
    if mfcc.shape[0] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_pad_len, :]
    return mfcc.reshape(1, max_pad_len, 40)

def predict_gender(file_path):
    model = tf.keras.models.load_model(r".\models\voice_gender_classifier.h5")
    features = preprocess_audio(file_path)
    prediction = model.predict(features)
    predicted_class = labelencoder.inverse_transform([np.argmax(prediction)])
    return predicted_class[0]
    

def record_audio(duration=3, sample_rate=22050, filename="temp.wav"):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    st.write("Recording...")
    frames = []
    for _ in range(0, int(sample_rate / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    st.write("Finished recording")
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return filename

path=st.text_input("Enter path of file")
if path:
    result=predict_gender(path)
    st.write(result)

live_chat_button=st.button(label="live",icon="🎙️")
if live_chat_button:
    audio=record_audio(duration=5)
    result=predict_gender(audio)
    st.write(f"voice : {result}")


    



