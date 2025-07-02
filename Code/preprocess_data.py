import pandas as pd
import os 
from preproccesing_methods import clean_text, add_sentiment_column

#Loads raw data
df = pd.read_csv('Data/WELFake_Dataset.csv')

#Drops rows with NA values in the title and text columns
df = df.dropna(subset = ['title', 'text'])

#Renames Unnamed: 0 column for clarity
df.rename(columns = {'Unnamed: 0': 'article_ID'}, inplace= True)

print('Initializing data cleaning \n')

#Applies the creates clean title and text columns using the clean_text function
df['clean_title'] = df['title'].apply(clean_text)
df['clean_text'] = df['text'].apply(clean_text)

#Creates a sentiment column based on the text column
add_sentiment_column(df)

#Saves clean data to the data folder for further use
df.to_csv('Data/cleaned_news_dataset.csv', index = False)

print('\n')

if os.path.exists('Data/cleaned_news_dataset.csv'):
    print('✅ Cleaned data successfully saved to data folder')
    print('-------------------------------------------------\n')
    print('File name: cleaned_news_dataset.csv')
    print(f'Number of rows saved to csv: {len(df)}')
    print(f'Columns saved to csv: {list(df.columns)}')
else:
    print('❌ Data unsuccessfully saved, please rerun code')


