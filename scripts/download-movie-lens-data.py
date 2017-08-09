import io
import os
import requests
import zipfile

data_dir = './data/'

def download_and_extract_zip(url, folder):
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(folder)
    
if not os.path.exists(os.path.join(data_dir, 'ml-latest')):
    print("Downloading MovieLens latest dataset")
    download_and_extract_zip('http://files.grouplens.org/datasets/movielens/ml-latest.zip', data_dir)