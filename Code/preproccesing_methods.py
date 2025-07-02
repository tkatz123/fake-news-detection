import re
from nltk.sentiment import SentimentIntensityAnalyzer
import swifter


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