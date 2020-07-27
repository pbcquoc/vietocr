import yaml
from vietocr.tool.utils import download_config

url_config = {
        'vgg_transformer':'1TF8effeufpgkHqQFlmNWKsQtCMfDiooa',
        'resnet_transformer':'1GGhQqtMz4WloBh38U4sMlzLN6cpw5iag',
        'resnet_fpn_transformer':'1I3-m8wfVpsro1c3UupwxW97MYmP5evvh',
        'base':'1xiw7ZnT3WH_9HXoGpLbhW-m2Sm2nlthi',
        }

class Cfg(dict):
    def __init__(self, config_dict):
        super(Cfg, self).__init__(**config_dict)
        self.__dict__ = self

    @staticmethod
    def load_config_from_file(fname):
        base_config = download_config(url_config['base'])

        with open(fname, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        config.update(base_config)

        return Cfg(config)

    @staticmethod
    def load_config_from_name(name):
        base_config = download_config(url_config['base'])
        config = download_config(url_config[name])

        config.update(base_config)

        return Cfg(config)

    def save(self, fname):
        with open(fname, 'w') as outfile:
            yaml.dump(dict(self), outfile, default_flow_style=False, allow_unicode=True)

