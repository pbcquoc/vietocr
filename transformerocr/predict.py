import argparse
from PIL import Image

from transformerocr.tool.detector import TextDetector
from transformerocr.tool.config import Cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='foo help')
    parser.add_argument('--config', required=True, help='foo help')

    args = parser.parse_args()
    config = Cfg(args.config)

    detector = TextDetector(config)

    img = Image.open(args.img)
    s = detector.predict(img)

    print(s)

if __name__ == '__main__':
    main()
