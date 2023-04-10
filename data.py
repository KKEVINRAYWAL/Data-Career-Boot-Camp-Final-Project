import os
import json
import logging
import subprocess
import zipfile
from typing import Dict, Union



def extract_files(zip_file_path, extract_dir_path):
    """
    Extracts files from a zip archive.

    :param zip_file_path: The path to the zip archive.
    :param extract_dir_path: The directory to extract the files to.
    :return: True if the files were extracted successfully, False otherwise.
    """
    logger = logging.getLogger(__name__)
    logger.info('Extracting files from archive...')
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir_path)
        return True
    except Exception as e:
        logger.error(f'Error extracting files from archive: {e}')
        return False


def load_kaggle_api_credentials(creds_file_path: str = "kaggle.json") -> Union[Dict[str, str], None]:
    """
    Loads Kaggle API credentials from a JSON file.

    :param creds_file_path: The path to the Kaggle API credentials file. Defaults to "kaggle.json".
    :return: A dictionary containing the Kaggle API credentials, or None if an error occurred.
    """
    try:
        with open(creds_file_path, 'r') as f:
            kaggle_creds = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logging.error(f'Could not load Kaggle API credentials from {creds_file_path}.')
        return None

    return kaggle_creds


def set_kaggle_api_credentials(username, key):
    """
    Sets the Kaggle API credentials in the environment variables.

    :param username: The Kaggle API username.
    :param key: The Kaggle API key.
    """
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key


def download_dataset(dataset_name: str, download_dir: str = "./data") -> bool:
    """
    Downloads and extracts the specified dataset from Kaggle.

    :param dataset_name: The name of the dataset to download.
    :param download_dir: The directory to save the downloaded dataset to. Defaults to "./data".
    :return: True if the dataset was downloaded successfully, False otherwise.
    """
    # Load Kaggle API credentials
    kaggle_creds = load_kaggle_api_credentials()
    if not kaggle_creds:
        return False

    # Set Kaggle API credentials
    set_kaggle_api_credentials(kaggle_creds['username'], kaggle_creds['key'])

    # Create a directory to store the downloaded data
    os.makedirs(download_dir, exist_ok=True)

    # Download the dataset
    logger = logging.getLogger(__name__)
    logger.info(f'Downloading dataset {dataset_name}...')
    try:
        output = subprocess.check_output(f'kaggle datasets download -d {dataset_name} -p {download_dir} --force', shell=True)
        logger.debug(output.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        logger.error(f'Error downloading dataset: {e}')
        return False

    # Extract the downloaded files
    zip_file_path = os.path.join(download_dir, f'{dataset_name}.zip')
    extract_dir_path = os.path.join(download_dir, dataset_name)
    logger = logging.getLogger(__name__)
    logger.info(f'Extracting files from {zip_file_path} to {extract_dir_path}...')
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir_path)
    except Exception as e:
        logger.error(f'Error extracting files from archive: {e}')
        return False

    return True


def get_animals_detection_images_dataset():
    """
    Downloads and extracts the animals detection images dataset from Kaggle.

    :return: The path to the directory containing the dataset, or None if an error occurred.
    """
    # Load Kaggle API credentials from kaggle.json file
    module_dir = os.path.dirname(os.path.abspath(__file__))
    creds_file = os.path.join(module_dir, 'kaggle.json')
    kaggle_creds = load_kaggle_api_credentials(creds_file)
    if not kaggle_creds:
        return None

    # Set Kaggle API credentials
    set_kaggle_api_credentials(kaggle_creds['username'], kaggle_creds['key'])

    # Create a directory to store the downloaded data
    dest_dir = os.path.join(module_dir, 'data')
    os.makedirs(dest_dir, exist_ok=True)

    # Download and extract the dataset
    dataset_path = download_dataset('antoreepjana/animals-detection-images-dataset', dest_dir)
    if not dataset_path:
        return None

    return dataset_path


   
