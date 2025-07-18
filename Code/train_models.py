import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from scipy.sparse import hstack
import joblib


def load_and_prepare_data(data):
    """
    Loads and prepares the dataset for model training.

    Combines the 'title' and 'text' columns into a single 'combined' column.
    Also fills any missing values in title or text with empty strings to avoid NaNs.

    Parameters
    ----------
    filepath : str
        Path to the cleaned CSV file.

    Returns
    -------
    pandas.DataFrame
        A dataframe with a new 'combined' text column.
    """

    #Checks if data is being passed as a filepath or dataframe
    if isinstance(data, str):
        #Reads in csv file from designated filepath
        df = pd.read_csv(data)
    else:
        df = data

    #Creates one string 'combined' from a combination of clean_title and clean_text
    df['combined'] = (df['clean_title'] + " " + df['clean_text']).fillna("")

    #Returns dataframe with new combined column
    return df

def split_data(df):
    """
    Splits the dataframe into training and testing sets for text, sentiment, and labels.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe with 'combined' and 'sentiment' columns.

    Returns
    -------
    tuple
        (X_train_text, X_test_text, X_train_sent, X_test_sent, y_train, y_test)
    """
    #Creates a series with the value of the combined columns
    X_text = df['combined']

    #Creates a dataframe with just the text_sentiment column needed for combining with sparse matrix later
    X_sent = df[['text_sentiment']]

    #Creates a serieis with the values of the label column
    y = df['label']

    #Returns split data 80% training, 20% testing, and stratifying the y column to ensure an equal balance of real/fake labels
    return train_test_split(X_text, X_sent, y, test_size = 0.2, stratify = y , random_state = 42)

def vectorize_text(X_train_text, X_test_text, max_features = 5000):
    """
    Fits a TF-IDF vectorizer on the training text and transforms both train and test sets.

    Parameters
    ----------
    X_train_text : array-like
        Training text samples.
    X_test_text : array-like
        Test text samples.
    max_features : int, optional
        Maximum number of features to retain in the TF-IDF vocabulary, by default 5000.

    Returns
    -------
    tuple
        (fitted TfidfVectorizer, X_train_tfidf, X_test_tfidf)
    """
    #Initializes a tfidf vectorizer with max_features passed in parameter
    tfidf = TfidfVectorizer(stop_words= 'english', max_features = max_features)

    #Fits and tranforms training data using tfidf vectorizer 
    X_train_tfidf = tfidf.fit_transform(X_train_text)

    #Transforms test data using tfidf vectorizer fitted on training data
    X_test_tfidf = tfidf.transform(X_test_text)

    #Returns tfidf vectorizer, and transformed train and testing data
    return tfidf, X_train_tfidf, X_test_tfidf

def combine_features(tfidf_matrix, sentiment_column):
    """
    Combines the TF-IDF matrix and sentiment column into a single feature matrix.

    Parameters
    ----------
    tfidf_matrix : scipy.sparse matrix
        The TF-IDF feature matrix.
    sentiment_column : pandas.DataFrame
        The sentiment score column (must be 2D).

    Returns
    -------
    scipy.sparse matrix
        Combined feature matrix for model input.
    """

    #Combines the tfidf matrix and sentiment column into one feature matrix for model training
    return hstack([tfidf_matrix, sentiment_column.values])

def train_models(X_train_combined, y_train):
    """
    Trains base models: logistic regression and random forest.

    Parameters
    ----------
    X_train_combined : scipy.sparse matrix
        Combined feature matrix of TF-IDF and sentiment.
    y_train : array-like
        Training labels.

    Returns
    -------
    tuple
        (trained LogisticRegression model, trained RandomForestClassifier model)
    """
    #Initializes a logistic regression model
    lr = LogisticRegression(max_iter = 1000, C = 10)

    #Initializes a random forest classifier model
    rf = RandomForestClassifier(max_depth = 15, max_features = "sqrt", min_samples_leaf = 1, min_samples_split = 5, n_estimators = 200)

    #Fits logistic regression model on training data
    lr.fit(X_train_combined, y_train)

    #Fits random forest classifier model on training data
    rf.fit(X_train_combined, y_train)

    #Returns fitted logistic regression and random forest classifier models 
    return lr, rf

def train_meta_model(lr, rf, X_train_combined, y_train):
    """
    Trains a meta-classifier using the predicted probabilities from
    logistic regression and random forest as input features.

    Parameters
    ----------
    lr : trained LogisticRegression
        The base logistic regression model.
    rf : trained RandomForestClassifier
        The base random forest model.
    X_train_combined : scipy.sparse matrix
        Feature matrix used for base model predictions.
    y_train : array-like
        True training labels.

    Returns
    -------
    LogisticRegression
        Trained meta-classifier model.
    """
    #Generate meta-features: predicted probabilities from base models (for class 1 = Real)
    meta_X = np.column_stack([
        lr.predict_proba(X_train_combined)[:, 1],  
        rf.predict_proba(X_train_combined)[:, 1] 
    ])

    #Train meta-classifier using logistic regression model
    meta = LogisticRegression()
    meta.fit(meta_X, y_train)

    #Return the trained meta-model
    return meta

def save_elements(tfidf, lr, rf, meta):
    """
    Saves trained models and the vectorizer to disk using joblib.

    Parameters
    ----------
    tfidf : TfidfVectorizer
        The fitted TF-IDF vectorizer.
    lr : LogisticRegression
        Trained logistic regression model.
    rf : RandomForestClassifier
        Trained random forest model.
    meta : LogisticRegression
        Trained meta-classifier model.

    Returns
    -------
    None
    """
    #Saves tfidf, linear regression, random forest classifier, and meta models as pkl files for importing later
    joblib.dump(tfidf, "Models/tfidf_vectorizer.pkl")
    joblib.dump(lr, "Models/lr_model.pkl")
    joblib.dump(rf, "Models/rf_model.pkl")
    joblib.dump(meta, "Models/meta_classifier.pkl")


def main():
    """
    Coordinates the model training workflow:
    - Loads and cleans data
    - Splits into training/testing sets
    - Vectorizes text
    - Combines features
    - Trains base and meta models
    - Saves artifacts to disk

    Returns
    -------
    None
    """
    #Loads data by calling load_and_prepare_data function
    df = load_and_prepare_data("Data/cleaned_news_dataset.csv")

    #Creates training and test splits for the text data, sentiment, and output labels using split_data
    X_train_text, X_test_text, X_train_sent, X_test_sent, y_train, y_test = split_data(df)

    #Creates vectorizor, which is used to create tfidf matrixes for train and test data using vectorize_text
    tfidf, X_train_tfidf, X_test_tfidf = vectorize_text(X_train_text, X_test_text)

    #Combines training tfidf matrix with training sentiment column to create one combined training matrix using combined_features
    X_train_combined = combine_features(X_train_tfidf, X_train_sent)

    #Trains logistic regression model and random forest classification model on training data using train_models
    lr_model, rf_model = train_models(X_train_combined, y_train)

    #Trains a meta classifier model (based on a logistic regression model) based on the results of the previous models to predict confidence
    #Using train_meta_model
    meta_clf = train_meta_model(lr_model, rf_model, X_train_combined, y_train)

    #Saves tfidf vectorizer, logistic regression, random forest classifier, and meta classification model using save_elements
    save_elements(tfidf, lr_model, rf_model, meta_clf)

    #Combines test text data, and sentiment data into a single feature matrix using combine features
    X_test_combined = combine_features(X_test_tfidf, X_test_sent)

    #Makes predictions using the logisitc regression and random forest classifier models based on combined testing data
    lr_preds = lr_model.predict(X_test_combined)
    rf_preds = rf_model.predict(X_test_combined)

    #Creates a matrix of confidence of real classifier for logisitc regression and random forest classifier models based on testing data
    meta_test_X = np.column_stack([
    lr_model.predict_proba(X_test_combined)[:, 1],
    rf_model.predict_proba(X_test_combined)[:, 1]
    ])

    #Makes a prediction using meta classifier model using testing matrix created previously
    meta_preds = meta_clf.predict(meta_test_X)
    
    #Making predictions using the training data
    lr_train_preds = lr_model.predict(X_train_combined)
    rf_train_preds = rf_model.predict(X_train_combined)

    #Retrieving f1 score per class of LR model
    lr_f1_per_class = f1_score(y_test, lr_preds, average = None)

    #Printing models accuracy on training data
    print("LR train accuracy: ", accuracy_score(y_train, lr_train_preds))
    print("RF train accuracy: ", accuracy_score(y_train, rf_train_preds))

    #Printing models accuracy on testing data
    print("LR test accuracy: ", accuracy_score(y_test, lr_preds))
    print("RF test accuracy: ", accuracy_score(y_test, rf_preds))

    #Print LR models f1 score for each class
    print(f"LR F1 Score (Fake): {lr_f1_per_class[0]:.4f}")
    print(f"LR F1 Score (Real): {lr_f1_per_class[1]:.4f}")

    # Logistic regression Evaluation
    print("Logistic Regression:")
    print("Accuracy:", accuracy_score(y_test, lr_preds))
    print(classification_report(y_test, lr_preds, target_names=["Fake (0)", "Real (1)"]))
    print(confusion_matrix(y_test, lr_preds))

    # Random forest Evaluation
    print("\nRandom Forest:")
    print("Accuracy:", accuracy_score(y_test, rf_preds))
    print(classification_report(y_test, rf_preds, target_names=["Fake (0)", "Real (1)"]))
    print(confusion_matrix(y_test, rf_preds))

    #Meta classifier evaluation
    print("\nClassification Report (Meta Classifier):")
    print("Accuracy:", accuracy_score(y_test, meta_preds))
    print(classification_report(y_test, meta_preds, target_names=["Fake (0)", "Real (1)"]))
    print(confusion_matrix(y_test, meta_preds))

    '''
    #Evaluating what the best value of C is for logistic regression model
    lr_param_grid = {"C": [0.001, 0.01, 0.1, 1, 10]}
    lr_grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
    lr_grid.fit(X_train_combined, y_train)

    print("Best C:", lr_grid.best_params_)
    print("Validation Score:", lr_grid.best_score_)
    '''

    '''
    #Determines optimal parameters for tuning RF model
    #Only run if needed, long run time
    rf_param_grid = {
    'max_depth': [10, 20, None],
    'min_samples_leaf': [1, 5, 10],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2'],
    'n_estimators': [100, 200]
    }

    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_param_grid,
        cv=3,
        scoring='f1',
        verbose=2,
        n_jobs=-1
    )

    rf_grid.fit(X_train_combined, y_train)

    print("Best parameters:", rf_grid.best_params_)
    print("Best CV score:", rf_grid.best_score_)
    '''
if __name__=="__main__":
    main()