import gdown
import yaml

def download_weights(id, md5, cached, quiet=False):
    url = 'https://drive.google.com/uc?id={}'.format(id)
    gdown.cached_download(url, cached, md5, quiet=quiet)

def download_config(id):
    url = 'https://drive.google.com/uc?id={}'.format(id)
    output = gdown.download(url, quiet=True)
    
    with open(output) as f:
        config = yaml.safe_load(f)

    return config
