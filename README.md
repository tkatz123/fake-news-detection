# Fake News Detection Using NLP & Streamlit

This project leverages **Natural Language Processing (NLP)** and machine learning to determine whether an article is **real (credible)** or **fake (misinformation)**. It includes a user-friendly **Streamlit interface** that allows users to input article text and receive predictions in real-time. The model highlights influential words and provides confidence scores for each prediction.

> If converted into a browser extension, this tool could inform readers of misinformation as they browse the internet.

---

## Requirements

This project was developed using **Python 3.11**. 

You can install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## How to Run the Application

To launch the Streamlit interface, enter the following command in your terminal:

```bash
streamlit run Code/app.py
```

This will start a local web server where you can interact with the fake news detection tool through a user friendly interface.

---

## Downloading the Dataset

Downloading the dataset is not required to use the core fake news detection algorithm via the Streamlit interface. However, it is required if you intend to run the accompanying Jupyter notebooks or scripts for exploration or model retraining. (Note: Depending on your system, the download and extraction process may take a few minutes.)

You can download the dataset using one of the following methods:

- **Via Streamlit Interface**

    Launch the app and navigate to the “Download & Preprocess Data” page using the sidebar navigation. Upload your `kaggle.json` API key file when prompted. The dataset will be downloaded and extracted automatically. Once the dataset is downloaded, it will be preprocessed and saved as a new file.

- **Via Manual Script Execution**

    Open download_data.py, scroll to the section under if __name__ == "__main__":, and manually enter your Kaggle username and API key in the designated fields, then run the file. Next open preprocessing_methods.py and run the script.

 Note: You can generate your Kaggle API key by creating a free account at the [Kaggle website](https://www.kaggle.com/) and following the instructions under Account Settings → Create API Token. This will download a kaggle.json file containing your credentials.

---

## Features

- Detects fake vs. real articles using multiple machine learning models
- Meta-classifier that combines Logistic Regression and Random Forest predictions
- Confidence scores for both fake and real classification
- Sentiment analysis integration (via NLTK VADER)
- TF-IDF-based keyword weighting
- Dataset download support via user-uploaded Kaggle API key

---

## Project Structure

- `Code/app.py`: Main Streamlit application for user interaction and real-time predictions
- `Code/download_data.py`: Script to securely download the dataset using a user-provided Kaggle API key
- `Code/exploratory_analysis.ipynb`: Jupyter notebook containing preliminary exploratory data analysis of the dataset
- `Code/preprocessing_methods.ipynb`: Functions for text preprocessing, cleaning, and sentiment annotation
- `Code/preprocessing_notebook.ipynb`: Development notebook used for iterative testing of preprocessing techniques
- `Code/train_models.py`: Pipeline for training, evaluating, and saving Logistic Regression, Random Forest, and meta-classifier models
- `Code/`: Directory used for organizing script files and notebooks
- `Models/`: Output directory containing saved trained models and TF-IDF vectorizer
- `Data/`: Directory for storing the raw and processed dataset after download; automatically created upon execution of download_data.py

---

## Models Used

### 1. Logistic Regression
- Linear model
- Used for classification and interpretation of keyword influence

### 2. Random Forest
- Nonlinear ensemble model
- More robust and better generalization

### 3. Meta Classifier
- Logistic Regression that takes the output probabilities from the two models above
- Produces a combined prediction and confidence score


Note: if you retune any of the model parameters, return the `train_models.py` script before running the Streamlit interface.
---

## Packages Used

- kaggle
- pandas
- matplotlib
- scikit-learn
- wordcloud
- nltk
- swifter
- scipy
- joblib

---

## License

This projest is licensed under the MIT Licesne. See the LICESNE for details.

