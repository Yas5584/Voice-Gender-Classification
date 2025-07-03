import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import os
import pickle
MODEL_DIR = "../models"

os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR,"labelencoder.pkl"),"rb") as f:
    labelencoder = pickle.load(f)
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
    model = tf.keras.models.load_model(r"..\models\voice_gender_classifier.h5")
    features = preprocess_audio(file_path)
    prediction = model.predict(features)
    print(prediction)
    predicted_class = labelencoder.inverse_transform([np.argmax(prediction)])
    # print("Predicted Gender:", predicted_class[0])
    return predicted_class[0]

