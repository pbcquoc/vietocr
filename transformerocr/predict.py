import yaml
import argparse
from PIL import Image

from transformerocr.tool.detector import TextDetector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='foo help')
    parser.add_argument('--config', default='config/vgg-transformer.yml', help='foo help')

    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    
    detector = TextDetector(config)

    img = Image.open(args.img)
    s = detector.predict(img)

    print(s)

if __name__ == '__main__':
    main()
