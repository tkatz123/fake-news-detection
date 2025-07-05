import re
from nltk.sentiment import SentimentIntensityAnalyzer
import swifter
import pandas as pd
import os


#Converted all cleaning code to use re.sub() so the function can be used with the apply method for reuse later with the streamlit
def clean_text(text):
    '''
    Cleans input text by:
    - Converting text to lowercase
    - Removing content in parenthesis
    - Removing numbers
    - Removing single-character words
    - Normalizing whitespace
    '''
    #Converts all text to lowercase
    text = text.lower()

    #Removes text within parenthesis
    text = re.sub(r'\([^)]*\)', '', text)

    #Keeps only characters and whitespace removing punctiation
    text = re.sub(r'[^\w\s]', '', text)

    #Removes any numerical values from the text
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    #Removes all single character words
    text = re.sub(r'\b\w\b', '', text)

    #Removes unnecessary internal spacing and internal/external spacing
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def add_sentiment_column(df, text_column='text'):
    """
    Adds a sentiment score column to a DataFrame using VADER sentiment analysis.

    This function computes the compound sentiment score of the specified text column
    and adds the results as a new column called 'text_sentiment'.

    Args:
        df (pd.DataFrame): The input DataFrame containing the text data.
        text_column (str, optional): The name of the column containing text to analyze.
            Defaults to 'text'.

    Returns:
        None: The function modifies the DataFrame in-place by adding the new column.
    """
    sia = SentimentIntensityAnalyzer()
    df['text_sentiment'] = df[text_column].fillna('').swifter.apply(lambda t: sia.polarity_scores(t)['compound'])

def preproccess_data(filepath = 'Data/WELFake_Dataset.csv'):
    #Loads raw data
    df = pd.read_csv(filepath)

    #Drops rows with NA values in the title and text columns
    df = df.dropna(subset = ['title', 'text'])

    #Renames Unnamed: 0 column for clarity
    df.rename(columns = {'Unnamed: 0': 'article_ID'}, inplace= True)

    #Applies the creates clean title and text columns using the clean_text function
    df['clean_title'] = df['title'].apply(clean_text)
    df['clean_text'] = df['text'].apply(clean_text)

    #Creates a sentiment column based on the text column
    add_sentiment_column(df)

    #Saves clean data to the data folder for further use
    df.to_csv('Data/cleaned_news_dataset.csv', index = False)

    return df

if __name__ == "__main__":

    print('Initializing data cleaning \n')

    df = preproccess_data()

    print('\n')

    if os.path.exists('Data/cleaned_news_dataset.csv'):
        print('✅ Cleaned data successfully saved to data folder')
        print('-------------------------------------------------\n')
        print('File name: cleaned_news_dataset.csv')
        print(f'Number of rows saved to csv: {len(df)}')
        print(f'Columns saved to csv: {list(df.columns)}')
    else:
        print('❌ Data unsuccessfully saved, please rerun code')