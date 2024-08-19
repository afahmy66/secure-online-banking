import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import time

# Define function to extract MFCC features
def extract_features(file_path):
    sample_rate, audio = wavfile.read(file_path)
    mfcc_features = mfcc(audio, sample_rate)
    return mfcc_features