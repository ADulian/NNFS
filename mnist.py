import os
import cv2
import numpy as np
import urllib
import urllib.request
from zipfile import ZipFile


def create_mnist_dataset(path):
    X, y = load_mnist_dataset('train',path)
    X_test, y_test = load_mnist_dataset('test',path)

    return X, y, X_test, y_test

def load_mnist_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))
    X, y = [], []
    for label in labels:
        for file in os.listdir(os.path.join('fashion_mnist_images', dataset, label)):
            X.append(cv2.imread(os.path.join('fashion_mnist_images', dataset, label, file), cv2.IMREAD_UNCHANGED))
            y.append(label)

    return np.array(X), np.array(y).astype('uint8')

def download_data():
    URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
    FILE = 'fashion_mnist_images.zip'
    FOLDER = 'fashion_mnist_images'

    if not os.path.isfile(FILE):
        print(f'Downloading {URL} and saving as {FILE}...')
        urllib.request.urlretrieve(URL, FILE)

    print("Unzipping images...")
    with ZipFile(FILE) as zip_images:
        zip_images.extractall(FOLDER)

    print("Done")
