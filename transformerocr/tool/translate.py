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

            del output

        translated_sentence = np.asarray(translated_sentence).T
    
    return translated_sentence

def build_model(config):
    vocab = Vocab(config['vocab'])
    device = config['device']
    
    model = TransformerOCR(len(vocab), 
            ss=config['cnn']['pooling_stride_size'], ks=config['cnn']['pooling_kernel_size'], 
            **config['transformer'])
    
    model = model.to(device)

    return model, vocab

def process_input(image):
    img = image.convert('RGB')
    w, h = img.size
    new_w = int(32 * float(w) / float(h))
    img = img.resize((new_w, 32), Image.ANTIALIAS)

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

