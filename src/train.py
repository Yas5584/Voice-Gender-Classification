from preprocess import load_metadata
from features import feature_extractor
import os
from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM,Bidirectional
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping

MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)
DATA_PATH = r"..\data\cv-corpus-10.0-delta-2022-07-04\en\clips"
METADATA_PATH = r"..\data\cv-corpus-10.0-delta-2022-07-04\en\validated.tsv"




def feature_enginering(metadata,audio):
    extracted_features=[]
    max_pad_len = 100 
    metadata=load_metadata(metadata,audio)

    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
      try:
        mfcc = feature_extractor(row["path"])
        if mfcc.shape[0] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[0]
            mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:max_pad_len, :]
        extracted_features.append([mfcc, row["gender"]])
        
      except:
        continue
    return extracted_features
# BiLSTM model definition

def model():
   model = Sequential([
    Bidirectional(LSTM(128, return_sequences=False), input_shape=(100, 40)),
    Dropout(0.5),
    Dense(100, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
    ])

   model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )
   return model





def train_model():
  extracted_features=feature_enginering(METADATA_PATH,DATA_PATH) 
  extracted_df = pd.DataFrame(extracted_features, columns=['features', 'gender'])
  x = np.array(extracted_df['features'].to_list())  # shape: (samples, 100, 40)
  y_labels = np.array(extracted_df['gender'].to_list())
  labelencoder=LabelEncoder()

  y_encoded = labelencoder.fit_transform(y_labels)
  y = to_categorical(y_encoded)

  with open(os.path.join(MODEL_DIR,"labelencoder.pkl"),"wb") as f:
    labelencoder = pickle.dump(labelencoder,f)

  X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

  cw = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
  cw_dict = dict(enumerate(cw))
  model=model()
  
  history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=cw_dict,
    callbacks=[EarlyStopping(patience=3)]
)
  model.save(os.path.join(MODEL_DIR, "voice_gender_classifier1.h5"))
  y_pred = model.predict(X_test)
  y_pred_labels = np.argmax(y_pred, axis=1)
  y_true_labels = np.argmax(y_test, axis=1)
  print(classification_report(y_true_labels, y_pred_labels, target_names=labelencoder.classes_))
  



train_model()


