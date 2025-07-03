import pyaudio
import wave
import numpy as np
from predict import predict_gender,preprocess_audio

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
    
    print("Recording...")
    frames = []
    for _ in range(0, int(sample_rate / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("Finished recording")
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

if __name__ == "__main__":
    
       audio_file = record_audio(duration=5)
       features = preprocess_audio(audio_file)
       
       result = predict_gender(features)
       print(result)
       