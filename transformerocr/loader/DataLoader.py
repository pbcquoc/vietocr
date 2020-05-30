import torch
import numpy as np
from PIL import Image
import random
from transformerocr.model.vocab import Vocab
import os
from collections import defaultdict
import math

class BucketData(object):
    def __init__(self, device):
        self.max_label_len = 0
        self.data_list = []
        self.label_list = []
        self.file_list = []
        self.device = device

    def append(self, datum, label, filename):
        self.data_list.append(datum)
        self.label_list.append(label)
        self.file_list.append(filename)
        
        self.max_label_len = max(len(label), self.max_label_len)

        return len(self.data_list)

    def flush_out(self):                           
        """
        Shape:
            - img: (N, C, H, W) 
            - tgt_input: (T, N) 
            - tgt_output: (N, T) 
            - tgt_padding_mask: (N, T) 
        """
        # encoder part
        img = np.array(self.data_list)/255                     
        
        # decoder part
        target_weights = []
        tgt_input = []
        for label in self.label_list:
            label_len = len(label)
            
            tgt = np.concatenate((
                label,
                np.zeros(self.max_label_len - label_len, dtype=np.int32)))
            tgt_input.append(tgt)
            
            one_mask_len = label_len - 1
            
            target_weights.append(np.concatenate((
                np.ones(one_mask_len, dtype=np.float32),
                np.zeros(self.max_label_len - one_mask_len,dtype=np.float32))))

        # reshape to fit input shape
        tgt_input = np.array(tgt_input).T
        tgt_output = np.roll(tgt_input, -1, 0).T
        tgt_output[:, -1]=0
        
        tgt_padding_mask = np.array(target_weights)==0 

        filenames = self.file_list

        self.data_list, self.label_list, self.file_list = [], [], []
        self.max_label_len = 0
        
        rs = {
            'img':torch.FloatTensor(img).to(self.device),
            'tgt_input':torch.LongTensor(tgt_input).to(self.device),
            'tgt_output':torch.LongTensor(tgt_output).to(self.device),
            'tgt_padding_mask':torch.BoolTensor(tgt_padding_mask).to(self.device),
            'filenames': filenames
        }
        
        return rs

    def __len__(self):
        return len(self.data_list)

    def __iadd__(self, other):
        self.data_list += other.data_list
        self.label_list += other.label_list
        self.max_label_len = max(self.max_label_len, other.max_label_len)
        self.max_width = max(self.max_width, other.max_width)

    def __add__(self, other):
        res = BucketData()
        res.data_list = self.data_list + other.data_list
        res.label_list = self.label_list + other.label_list
        res.max_width = max(self.max_width, other.max_width)
        res.max_label_len = max((self.max_label_len, other.max_label_len))
        return res

class DataGen(object):

    def __init__(self,data_root, annotation_fn, vocab, device):
        
        self.image_height = 32
        self.data_root = data_root
        self.annotation_path = os.path.join(data_root, annotation_fn)
        
        self.vocab = vocab
        self.device = device
        
        self.clear()

    def clear(self):
        self.bucket_data = defaultdict(lambda: BucketData(self.device))

    def gen(self, batch_size, last_batch=True):
        with open(self.annotation_path, 'r') as ann_file:
            lines = ann_file.readlines()
            np.random.shuffle(lines)
            for l in lines:     
                
                img_path, lex = l.strip().split('\t')
                
                img_path = os.path.join(self.data_root, img_path)
                
                try:
                    img_bw, word = self.read_data(img_path, lex)
                except IOError:
                    print('ioread image:{}'.format(img_path))
                    
                width = img_bw.shape[-1]

                bs = self.bucket_data[width].append(img_bw, word, img_path)
                if bs >= batch_size:
                    b = self.bucket_data[width].flush_out()
                    yield b

        if last_batch: 
            for bucket in self.bucket_data.values():
                if len(bucket) > 0:
                    b = bucket.flush_out()
                    yield b

        self.clear()

    def read_data(self, img_path, lex):        
        
        with open(img_path, 'rb') as img_file:
            img = Image.open(img_file).convert('RGB')
            w, h = img.size
            new_w = int(self.image_height * float(w) / float(h))
            new_w = math.ceil(new_w/10)*10
            new_w = min(new_w, 500)
            img = img.resize((new_w, self.image_height), Image.ANTIALIAS)

            img_bw = np.asarray(img).transpose(2,0, 1)
            
        word = self.vocab.encode(lex)

        return img_bw, word

