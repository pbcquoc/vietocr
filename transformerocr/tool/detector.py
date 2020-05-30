from transformerocr.tool.translate import build_model, translate, process_input, predict
import yaml

class TextDetector():
    def __init__(self, config):
        
        device = config['device']

        model, vocab = build_model(config)
        
        model.load_state_dict(torch.load(config['weights'], map_location=torch.device(device)))

        self.config = config
        self.model = model
        self.vocab = vocab
        

    def predict(self, img):
        img = process_input(img)
        img = img.to(self.config['device'])

        s = translate(img, self.model)[0].tolist()
        s = self.vocab.decode(s)

        return s

