import os
import subprocess
import json
import zipfile
import nltk


#Modularized version of code above to use in streamlit
def download_data(kaggle_json_file):

    creds = json.load(kaggle_json_file)

    # Set environment variables
    os.environ['KAGGLE_USERNAME'] = creds["username"]
    os.environ['KAGGLE_KEY'] = creds["key"]

    # Replace with your actual dataset (e.g., 'zynicide/wine-reviews')
    dataset = 'saurabhshahane/fake-news-classification'
    download_dir = 'Data'

    # Create the data directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)

    # Use curl to call the Kaggle API for dataset download
    curl_cmd = f"""
    curl -L -u {'KAGGLE_USERNAME'}:{'KAGGLE_KEY'} https://www.kaggle.com/api/v1/datasets/download/{dataset} \
    -o {download_dir}/dataset.zip
    """

    # Run the curl command
    subprocess.run(curl_cmd, shell=True, check=True)

    #Unzips downloaded file
    with zipfile.ZipFile(f"{download_dir}/dataset.zip", 'r') as zip_ref:
        zip_ref.extractall(download_dir)

    print("✅ Dataset downloaded successfully.")

    # Ensures the VADER lexicon is available for sentiment analysis
    nltk.download('vader_lexicon', quiet = True)

if __name__ == "__main__":
    # Set environment variables
    os.environ['KAGGLE_USERNAME'] = 'ENTER YOUR KAGGLE USERNAME'
    os.environ['KAGGLE_KEY'] = 'ENTER YOUR API KEY'

    # Replace with your actual dataset (e.g., 'zynicide/wine-reviews')
    dataset = 'saurabhshahane/fake-news-classification'
    download_dir = 'Data'

    # Create the data directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)

    # Use curl to call the Kaggle API for dataset download
    curl_cmd = f"""
    curl -L -u {'KAGGLE_USERNAME'}:{'KAGGLE_KEY'} https://www.kaggle.com/api/v1/datasets/download/{dataset} \
    -o {download_dir}/dataset.zip
    """

    # Run the curl command
    subprocess.run(curl_cmd, shell=True, check=True)

    #Unzips downloaded file
    with zipfile.ZipFile(f"{download_dir}/dataset.zip", 'r') as zip_ref:
        zip_ref.extractall(download_dir)

    print("✅ Dataset downloaded successfully.")

    # Ensures the VADER lexicon is available for sentiment analysis
    nltk.download('vader_lexicon', quiet = True)





