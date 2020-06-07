import torch
import numpy as np
import math
from PIL import Image
from vietocr.model.transformerocr import VietOCR
from vietocr.model.vocab import Vocab

def translate(img, model, max_seq_length=128):
    "data: BxCXHxW"
    model.eval()
    device = img.device
    src = model.cnn(img)
    memory = model.transformer.forward_encoder(src)

    with torch.no_grad():
        translated_sentence = [[1]*len(img)]
        max_length = 0

        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T==2, axis=1)):

            tgt_inp = torch.LongTensor(translated_sentence).to(device)
            
#            output = model(img, tgt_inp, tgt_key_padding_mask=None)
#            output = model.transformer(src, tgt_inp, tgt_key_padding_mask=None)
            output = model.transformer.forward_decoder(tgt_inp, memory)
            output = output.to('cpu')

            values, indices  = torch.topk(output, 5)

            indices = indices[:, -1, 0]
            indices = indices.tolist()

            translated_sentence.append(indices)   
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T
    
    return translated_sentence

def build_model(config):
    vocab = Vocab(config['vocab'])
    device = config['device']
    
    model = VietOCR(len(vocab), 
            ss=config['cnn']['pooling_stride_size'], ks=config['cnn']['pooling_kernel_size'], 
            **config['transformer'])
    
    model = model.to(device)

    return model, vocab

def process_input(image):
    img = image.convert('RGB')
    w, h = img.size
    new_w = int(32 * float(w) / float(h))
    new_w = math.ceil(new_w/10)*10
    new_w = min(new_w, 500)
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

