import streamlit as st
import os
from download_data import download_data
from preproccesing_methods import preproccess_data

st.title('Download data')

#Checks to see if data has already been downloaded
if os.path.exists('Data/WELFake_Dataset.csv'):
    st.success('✅ Data already downloaded')

    if os.path.exists('Data/cleaned_news_dataset.csv'):
        st.success('✅ Data already cleaned')
        
#If data has not been downloaded, asks user to upload API key to download data
else:
    st.write('📂 File not found. Upload kaggle.json to download data')
    kaggle_json = st.file_uploader('Upload kaggle.json')

    #Downloads dataset from kaggle once API key is successfully uploaded
    if kaggle_json is not None:
        st.write("🔽 File uploaded, downloading data...")
        download_data(kaggle_json)

        #Once dataset is successfully downloaded, preproccesses data 
        if os.path.exists('Data/WELFake_Dataset.csv'):
            st.success('✅ Data downloaded successfully')

            st.write('Preprocessing data, please be patient...')

            preproccess_data()

            if os.path.exists('Data/cleaned_news_dataset.csv'):
                st.success('✅ Data preprocessing successful')
            
