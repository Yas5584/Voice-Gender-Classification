import pandas as pd
import os
import numpy as np

def load_metadata(data_path,audio_path):
    df = pd.read_csv(data_path, sep='\t')
    df = df[['path', 'gender']]
    df = df[(df['gender'] == 'male') | (df['gender'] == 'female')]
    df_labeled= df.dropna(subset=['gender'])
    df_unlabeled = df[df['gender'].isna()]

    df_labeled['path'] =audio_path + '/' + df_labeled['path']
    return df_labeled

