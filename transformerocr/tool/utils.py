import gdown

def download_weights(id, md5, cached, quiet=True):
    url = 'https://drive.google.com/uc?id={}'.format(id)
    gdown.cached_download(url, cached, md5, quiet=quiet)

