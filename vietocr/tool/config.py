import yaml
from vietocr.tool.utils import download_config

url_config = {
        'vgg_transformer':'1TF8effeufpgkHqQFlmNWKsQtCMfDiooa',
        'resnet_transformer':'1GGhQqtMz4WloBh38U4sMlzLN6cpw5iag',
        'resnet_fpn_transformer':'1I3-m8wfVpsro1c3UupwxW97MYmP5evvh',
        'vgg_seq2seq':'1lWUvdYnyZ6HI52I6THS_Zr97YwEzcROn',
        'vgg_convseq2seq':'1f5On-N-Dc25LZq0ZHLR3uhNlHVPkXl60',
        'vgg_decoderseq2seq':'10YrSoK_gFuuhTN_u6emOgYEu5v7Y4ksG',
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
        base_config.update(config)

        return Cfg(base_config)

    @staticmethod
    def load_config_from_name(name):
        base_config = download_config(url_config['base'])
        config = download_config(url_config[name])

        base_config.update(config)
        return Cfg(base_config)

    def save(self, fname):
        with open(fname, 'w') as outfile:
            yaml.dump(dict(self), outfile, default_flow_style=False, allow_unicode=True)

