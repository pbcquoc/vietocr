import yaml
import argparse

from tool.translate import build_model, translate, process_input, predict

class TextDetector():
    def __init__(self, config):
        model, vocab = build_model(config)

        self.config = config
        self.model = model
        self.vocab = vocab

    def predict(self, img):
        img = process_input(img)
        img = img.to(self.config['device'])

        s = translate(img, self.model)[0].tolist()
        s = vocab.decode(s)

        return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='foo help')
    parser.add_argument('--config', default='config.yml', help='foo help')

    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    s = predict(args.img, config)

    print(s)
