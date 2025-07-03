import streamlit as st
import os
from download_data import download_data

st.title('Download data')

#Checks to see if data has already been downloaded
if os.path.exists('Data/WELFake_Dataset.csv'):
    st.write('✅ Data already downloaded')
#If data has not been downloaded, asks user to upload API key to download data
else:
    st.write('📂 File not found. Upload kaggle.json to download data')
    kaggle_json = st.file_uploader('Upload kaggle.json')

    if kaggle_json is not None:
        st.write("🔽 File uploaded, downloading data...")
        download_data(kaggle_json)

        if os.path.exists('Data/WELFake_Dataset.csv'):
            st.write('✅ Data downloaded successfully')
