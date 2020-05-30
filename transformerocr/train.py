import argparse

from transformerocr.model.trainer import Trainer
from transformerocr.tool.config import Cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='see example at ')
    parser.add_argument('--checkpoint', required=False, help='your checkpoint')

    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)

    trainer = Trainer(config)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        
    trainer.train()

if __name__ == '__main__':
    main()
