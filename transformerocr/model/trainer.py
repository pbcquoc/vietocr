from transformerocr.optim.optim import ScheduledOptim
from torch.optim import Adam
from torch import nn
from transformerocr.tool.translate import build_model
from transformerocr.tool.translate import translate
from transformerocr.tool.utils import download_weights
from einops import rearrange
import yaml
import torch
from transformerocr.loader.DataLoader import DataGen
from transformerocr.tool.utils import compute_accuracy
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self, config, pretrain=True):

        self.config = config
        self.model, self.vocab = build_model(config)

        self.device = config['device']
        self.num_epochs = config['trainer']['epochs']
        self.data_root = config['trainer']['data_root']
        self.train_annotation = config['trainer']['train_annotation']
        self.valid_annotation = config['trainer']['valid_annotation']
        self.batch_size = config['trainer']['batch_size']
        self.print_every = config['trainer']['print_every']
        self.valid_every = config['trainer']['valid_every']
        self.checkpoint = config['trainer']['checkpoint']
        self.export_weights = config['trainer']['export']

        if pretrain:
            download_weights(**config['pretrain'], quiet=config['quiet'])
            self.model.load_state_dict(torch.load(config['pretrain']['cached'], map_location=torch.device(self.device)))

        self.epoch = 0 
        self.iter = 0

        self.optimizer = ScheduledOptim(
            Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
            0.2, config['transformer']['d_model'], config['optimizer']['n_warmup_steps'])

        self.criterion = nn.CrossEntropyLoss(ignore_index=0) 

        self.train_gen = DataGen(self.data_root, self.train_annotation, self.vocab, self.device)
        if self.valid_annotation:
            self.valid_gen = DataGen(self.data_root, self.valid_annotation, self.vocab, self.device)
        
        self.train_losses = []

    def train(self):
        total_loss = 0
       
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            for batch in self.train_gen.gen(self.batch_size, last_batch=False):
                self.iter += 1

                loss = self.step(batch)
                
                total_loss += loss
                self.train_losses.append((self.iter, loss))

                if self.iter % self.print_every == self.print_every - 1:
                    info = 'iter: {:06d} - epoch: {:03d} - train loss: {:.4f}'.format(self.iter, epoch, total_loss/self.print_every)
                    total_loss = 0
                    print(info) 
                
                if self.valid_annotation and self.iter % self.valid_every == self.valid_every - 1:
                    val_loss = self.validate()
                    acc_full_seq, acc_per_char = self.precision()

                    info = 'iter: {:06d} - epoch: {:03d} - val loss: {:.4f} - acc full seq: {:.4f} - acc per char: {:.4f}'.format(
                            self.iter, epoch, val_loss, 
                            acc_full_seq, acc_per_char)
                    print(info)
                    self.save_checkpoint(self.checkpoint)
                    self.save_weight(self.export_weights)
        
    def validate(self):
        self.model.eval()

        total_loss = []
        
        with torch.no_grad():
            for step, batch in enumerate(self.valid_gen.gen(self.batch_size)):

                img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch['tgt_input'], batch['tgt_output'], batch['tgt_padding_mask']

                outputs = self.model(img, tgt_input, tgt_padding_mask)

                loss = self.criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_output, 'b o -> (b o)'))

                total_loss.append(loss.item())
                
                del outputs
                del loss

        total_loss = np.mean(total_loss)
        self.model.train()
        
        return total_loss
    
    def predict(self, sample=None):
        pred_sents = []
        actual_sents = []
        img_files = []
        
        n = 0
        for batch in  self.valid_gen.gen(self.batch_size):
            translated_sentence = translate(batch['img'], self.model)
            pred_sent = self.vocab.batch_decode(translated_sentence.tolist())
            actual_sent = self.vocab.batch_decode(batch['tgt_input'].T.tolist())

            img_files.extend(batch['filenames'])

            pred_sents.extend(pred_sent)
            actual_sents.extend(actual_sent)
            n += len(actual_sents)
            
            if sample != None and n > sample:
                break

        return pred_sents, actual_sents, img_files

    def precision(self):

        pred_sents, actual_sents, _ = self.predict(sample=None)

        acc_full_seq = compute_accuracy(actual_sents, pred_sents, mode='full_sequence')
        acc_per_char = compute_accuracy(actual_sents, pred_sents, mode='per_char')
    
        return acc_full_seq, acc_per_char
    
    def visualize(self, sample=32):
        
        pred_sents, actual_sents, img_files = self.predict(sample)
        img_files = img_files[:sample]

        for vis_idx in range(0, len(img_files)):
            img_path = img_files[vis_idx]
            pred_sent = pred_sents[vis_idx]
            actual_sent = actual_sents[vis_idx]

            img = Image.open(open(img_path, 'rb'))
            plt.figure()
            plt.imshow(img)
            plt.title('pred: {} - actual: {}'.format(pred_sent, actual_sent), loc='left')
            plt.axis('off')

        plt.show()


    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        
        optim = ScheduledOptim(
            Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
            0.2, self.config['transformer']['d_model'], self.config['optimizer']['n_warmup_steps'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.epoch = checkpoint['epoch']
        self.iter = checkpoint['iter']

        self.train_losses = checkpoint['train_losses']

    def save_checkpoint(self, filename):
        state = {'iter':self.iter, 'epoch': self.epoch, 'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'train_losses': self.train_losses}
        
        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)

        torch.save(state, filename)

    
    def save_weight(self, filename):
        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)
       
        torch.save(self.model.state_dict(), filename)

    def step(self, batch):
        self.model.train()
        
        img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch['tgt_input'], batch['tgt_output'], batch['tgt_padding_mask']
        
        outputs = self.model(img, tgt_input, tgt_key_padding_mask=tgt_padding_mask)
        loss = self.criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_output, 'b o -> (b o)'))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step_and_update_lr()
        
        loss_item = loss.item()

        del outputs
        del loss

        return loss_item
