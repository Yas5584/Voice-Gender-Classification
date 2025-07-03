import librosa
import numpy as np


def feature_extractor(file):
    audio,sample_rate=librosa.load(file,res_type="Kaiser_fast")
    audio, _ = librosa.effects.trim(audio)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return  mfccs_features.T


