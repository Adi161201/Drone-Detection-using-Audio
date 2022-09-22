import mimetypes
import numpy as np 
import matplotlib.pyplot as plt
import streamlit as st
import IPython.display as ipd
import librosa
import librosa.display
from scipy.io import wavfile as wav
from IPython.display import Audio
from pydub import AudioSegment

import time
from keras.models import load_model

import warnings
warnings.filterwarnings('ignore')



st.write("""
# Drone audio Classification
This app classifies the **Drone audio** from other 
environmental noises :)
""")

# st.snow()
st.set_option('deprecation.showPyplotGlobalUse', False)
try:
    data = st.file_uploader("Upload an Audio file " , 
                        type=["wav", "mp3"])
    # st.write(data)
    
        
    #     data= sound.export(data.wav , format="wav")
    # st.write(data.name)

    if(data is None):
        st.write("No file uploaded yet :( ")

    else:
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)

        # Getting File Extension
        x , y = (data.name).split(".")
    
        if (y=='mp3'):
            st.write("Yes")
            data = AudioSegment.from_mp3(data)
            data
            audio1, sample_rate1 = librosa.load(data, res_type='kaiser_fast') 
            plt.plot(audio1)
            st.pyplot(audio1.all())

        else :   
            st.audio(data, format='audio/ogg')
            
            wave_sample_rate , wave_audio = wav.read(data)
            

            st.write(" #### Plot of audio file ")
            plt.plot(wave_audio)
            st.pyplot(wave_audio.all())



    # Extracting mfccs for file

    def features_extractor(file):
    
        audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
        
        librosa.display.waveshow(audio, sr=sample_rate)
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
        return mfccs_scaled_features


    model = load_model('audio_classification_with_SVM_notebook_model3.hdf5')

    def predict_sound_ann(filename):
        prediction_feature=features_extractor(filename)
        prediction_feature=prediction_feature.reshape(1,-1)

        predicted_label= model.predict(prediction_feature)

        # print(" Class 0 represents Audio is Unknown ")
        # print(" Class 1 represents Audio is of Drone ")
        # print("Probabilities of predicted classes are " , predicted_label)
        
        if(predicted_label[0][0] > 0.5):
            st.subheader(" \n Audio is : Unknown ")
            
        elif(0.5 < predicted_label[0][1]):
            st.subheader(" \n Audio is of : Drone ")
        
        else:
            st.subheader(" Can't Say anything ")

    predict_sound_ann(data)

except:
    pass




   
    






