import yaml

class Cfg(dict):
    def __init__(self, config_file):
        with open(config_file) as f:
            config = yaml.safe_load(f)
        super(Cfg, self).__init__(**config)
        self.__dict__ = self
    
    def save(self, fname):
        with open(fname, 'w') as outfile:
            yaml.dump(dict(self), outfile, default_flow_style=False, allow_unicode=True)

