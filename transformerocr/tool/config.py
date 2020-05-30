import yaml
from transformerocr.tool.utils import download_config

url_config = {'vgg_transformer':'1TF8effeufpgkHqQFlmNWKsQtCMfDiooa'}

class Cfg(dict):
    def __init__(self, config_dict):
        super(Cfg, self).__init__(**config_dict)
        self.__dict__ = self

    @staticmethod
    def load_config_from_file(fname):
        with open(fname) as f:
            config = yaml.safe_load(f)

        return Cfg(config)

    @staticmethod
    def load_config_from_name(name):
        config = download_config(url_config[name])
        return Cfg(config)

    def save(self, fname):
        with open(fname, 'w') as outfile:
            yaml.dump(dict(self), outfile, default_flow_style=False, allow_unicode=True)

