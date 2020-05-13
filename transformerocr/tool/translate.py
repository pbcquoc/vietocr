import torch
import numpy as np
from PIL import Image
from transformerocr.model.transformerocr import TransformerOCR
from transformerocr.model.vocab import Vocab

def translate(img, model):
    "data: BxCXHxW"
    model.eval()
    device = img.device
    
    with torch.no_grad():
        translated_sentence = [[1]*len(img)]

        while not all(np.any(np.asarray(translated_sentence).T==2, axis=1)):

            tgt_inp = torch.LongTensor(translated_sentence).to(device)
            
            output = model(img, tgt_inp, tgt_key_padding_mask=None)
            output = output.to('cpu')

            values, indices  = torch.topk(output, 5)

            indices = indices[:, -1, 0]
            indices = indices.tolist()

            translated_sentence.append(indices)    
        translated_sentence = np.asarray(translated_sentence).T
    
    model.train()
    
    return translated_sentence

def build_model(config):
    vocab = Vocab(config['vocab'])
    device = config['device']
    
    model = TransformerOCR(len(vocab), **config['transformer'])
    model.load_state_dict(torch.load(config['weights']['cached'], map_location=torch.device(device)))

    model = model.to(device)

    return model, vocab

def process_input(image):
    img = image.convert('RGB')
    img = np.asarray(img).transpose(2,0, 1)
    img = img/255
    img = img[np.newaxis, ...]
    img = torch.FloatTensor(img)
    return img

def predict(filename, config):
    img = Image.open(filename)
    img = process_input(img)

    img = img.to(config['device'])

    model, vocab = build_model(config)
    s = translate(img, model)[0].tolist()
    s = vocab.decode(s)
    
    return s

