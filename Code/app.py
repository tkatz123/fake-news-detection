import streamlit as st
import os
import pandas as pd
import numpy as np
from download_data import download_data
from preproccesing_methods import preproccess_data, clean_text, add_sentiment_column
from train_models import load_and_prepare_data, combine_features
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

#Defines the webpage title
st.set_page_config(page_title="Fake News Detector", layout="wide")

#Specifies the background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #d0ebff; /* Light blue background */
    }
    </style>
    """,
    unsafe_allow_html=True
)

#Data downloading and preproccessing page
def page1():
    st.title('🖨️Download & Preproccess Data')

    #Checks to see if data has already been downloaded
    if os.path.exists('Data/WELFake_Dataset.csv'):
        st.success('✅ Data already downloaded')

        #Checks to see if data has already been preproccessed, if not calls preproccesiing function
        if os.path.exists('Data/cleaned_news_dataset.csv'):
            st.success('✅ Data already cleaned')
        else:
            st.write('Preprocessing data, please be patient...')

            preproccess_data()

            st.success('✅ Data preprocessing successful')

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

#News article validity evaluator
def page2():
    st.title('📰🚨Fake News Detector')

    #Loads models and vectorizor fitted on training data
    tfidf = joblib.load('Models/tfidf_vectorizer.pkl')
    lr_model = joblib.load('Models/lr_model.pkl')
    rf_model = joblib.load('Models/rf_model.pkl')
    meta_clf = joblib.load('Models/meta_classifier.pkl')

    #Initializes a session state if it doesn't exist
    if 'input1' not in st.session_state:
        st.session_state.input1 = ""
    if 'input2' not in st.session_state:
        st.session_state.input2 = ""

    #Define a function to clear text area inputs
    def clear_inputs():
        st.session_state.input1 = ""
        st.session_state.input2 = ""

    #Creates text box to accept article title
    title = st.text_area('Enter title of article', value = st.session_state.input1, key='input1')

    #Creates text box to accept article text
    text = st.text_area('Enter content of article', value = st.session_state.input2, key='input2')

    if st.button('Evaluate article validity'):

        #Create a dictionary of the inputted title and text
        data = {
            'title': [title],
            'text': [text]
        }

        #Convert dictionary to pandas dataframe
        df = pd.DataFrame(data)

        #Applies text cleaning function to inputted title and text
        df['clean_title'] = df['title'].apply(clean_text)
        df['clean_text'] = df['text'].apply(clean_text)

        #Creates sentiment column based on article text
        add_sentiment_column(df)

        #Creates a combined column using cleaned title and text
        load_and_prepare_data(df)

        #Extracts combined column
        text_data = df['combined'].values

        #Extracts sentiment column as a df
        text_sent = df[['text_sentiment']]

        #Creates a tfidf matrix of words in text_data
        tfidf_matrix = tfidf.transform(text_data)

        #Combines tfidf matrix and sentiment value into one feature matrix
        text_sent_combined = combine_features(tfidf_matrix, text_sent)

        #Makes predictions using the logisitic regression and random forest classifier model based on input data
        lr_preds = lr_model.predict(text_sent_combined)
        rf_preds = rf_model.predict(text_sent_combined)

        #Creates a matrix of confidence of real classifier for logisitc regression and random forest classifier models based on input data
        meta_input = np.column_stack([
        lr_model.predict_proba(text_sent_combined)[:, 1],
        rf_model.predict_proba(text_sent_combined)[:, 1]
        ])

        #Makes a prediction using meta classifier model using matrix created previously
        meta_probs = meta_clf.predict_proba(meta_input)

        #Gets the confidence of a real and fake prediction
        real_confidence = meta_probs[:, 1]  
        fake_confidence = meta_probs[:, 0]

        #Intializes three columns for the prediction metrics
        col1, col2, col3 = st.columns(3)

        #Displays metrics for predictions, and percentage confidence of each prediction category
        with col1:
            st.metric(label="Prediction", value='Fake' if fake_confidence[0] >= 0.5 else 'Real')
        with col2:
            st.metric(label="Real Confidence", value=f"{real_confidence[0]:.2%}")
        with col3:
            st.metric(label="Fake Confidence", value=f"{fake_confidence[0]:.2%}")

    #Button to clear text box inputs
    st.button("Clear Input", on_click=clear_inputs)
        

def main():
    st.sidebar.header("Navigation")
    
    # Create a sidebar selection menu
    page = st.sidebar.selectbox("Go to", ("Fake News Detector", "Download & Preproccess Data"))
    
    # Display the selected page
    if page == "Download & Preproccess Data":
        page1()
    elif page == "Fake News Detector":
        page2()

# Run the main function
if __name__ == "__main__":
    main()
            
