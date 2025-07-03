import os
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s:')
list_of_files = [
    "data/cv-corpus-10.0-delta-2022-07-04",       # Folder containing audio files
    "data/processed/",

    "models/voice_gender_cnn.h5",
    "models/scaler.pkl",

    "notebooks/model.ipynb",

    "src/preprocess.py",
    "src/features.py",
    "src/train.py",
    "src/predict.py",

    "samples/test_male.wav",
    "samples/test_female.wav",

    "reports/training_metrics.png",
    "reports/sample_prediction.png",
    "reports/feature_distribution.png",

    "requirements.txt",
    "README.md",
    ".gitignore"
]


for filepath in list_of_files:
    filepath=Path(filepath)
    filedir,filename=os.path.split(filepath)
    
    if filedir!="":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating directory {filedir} for file {filename  }")
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,"w") as f:
            pass
            logging.info(f"Creating empty file:{filepath}")
    else:
        logging.info(f"{filename} is already exists")