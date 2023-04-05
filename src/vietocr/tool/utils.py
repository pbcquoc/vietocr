import os
import gdown
import yaml
import numpy as np
import uuid
import requests
import tempfile
from tqdm import tqdm

def download_weights(uri, cached=None, md5=None, quiet=False):
    if uri.startswith('http'):
        return download(url=uri, quiet=quiet)
    return uri

def download(url, quiet=False):
    tmp_dir = tempfile.gettempdir()
    filename = url.split('/')[-1]
    full_path = os.path.join(tmp_dir, filename)
    
    if os.path.exists(full_path):
        print('Model weight {} exsits. Ignore download!'.format(full_path))
        return full_path

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(full_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk:
                f.write(chunk)
    return full_path

def download_config(id):
    url = 'https://vocr.vn/data/vietocr/config/{}'.format(id)
    r = requests.get(url)
    config = yaml.safe_load(r.text)
    return config

def compute_accuracy(ground_truth, predictions, mode='full_sequence'):
    """
    Computes accuracy
    :param ground_truth:
    :param predictions:
    :param display: Whether to print values to stdout
    :param mode: if 'per_char' is selected then
                 single_label_accuracy = correct_predicted_char_nums_of_single_sample / single_label_char_nums
                 avg_label_accuracy = sum(single_label_accuracy) / label_nums
                 if 'full_sequence' is selected then
                 single_label_accuracy = 1 if the prediction result is exactly the same as label else 0
                 avg_label_accuracy = sum(single_label_accuracy) / label_nums
    :return: avg_label_accuracy
    """
    if mode == 'per_char':

        accuracy = []

        for index, label in enumerate(ground_truth):
            prediction = predictions[index]
            total_count = len(label)
            correct_count = 0
            try:
                for i, tmp in enumerate(label):
                    if tmp == prediction[i]:
                        correct_count += 1
            except IndexError:
                continue
            finally:
                try:
                    accuracy.append(correct_count / total_count)
                except ZeroDivisionError:
                    if len(prediction) == 0:
                        accuracy.append(1)
                    else:
                        accuracy.append(0)
        avg_accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
    elif mode == 'full_sequence':
        try:
            correct_count = 0
            for index, label in enumerate(ground_truth):
                prediction = predictions[index]
                if prediction == label:
                    correct_count += 1
            avg_accuracy = correct_count / len(ground_truth)
        except ZeroDivisionError:
            if not predictions:
                avg_accuracy = 1
            else:
                avg_accuracy = 0
    else:
        raise NotImplementedError('Other accuracy compute mode has not been implemented')

    return avg_accuracy
