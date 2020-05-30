from transformerocr.optim.optim import ScheduledOptim
from torch.optim import Adam
from torch import nn
from transformerocr.tool.translate import build_model
from transformerocr.tool.utils import download_weights
from einops import rearrange
import yaml
import torch
from transformerocr.loader.DataLoader import DataGen
import numpy as np

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
        self.optimizer = ScheduledOptim(
            Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
            0.2, config['transformer']['d_model'], config['optimizer']['n_warmup_steps'])

        self.criterion = nn.CrossEntropyLoss(ignore_index=0) 

        self.train_gen = DataGen(self.data_root, self.train_annotation, self.vocab, self.device)
        if self.valid_annotation:
            self.valid_gen = DataGen(self.data_root, self.valid_annotation, self.vocab, self.device)
        
        self.train_losses = []

    def train(self):

        for epoch in range(self.num_epochs):
            total_loss = 0
            self.epoch = epoch
            for idx, batch in enumerate(self.train_gen.gen(self.batch_size)):
                        
                loss = self.step(batch)
                
                total_loss += loss
                self.train_losses.append((idx, loss))

                if idx % self.print_every == self.print_every - 1:
                    info = 'epoch: {} iter: {} - train loss: {}'.format(epoch, idx, total_loss/self.print_every)
                    total_loss = 0
                    print(info) 
                
                if idx % self.valid_every == self.valid_every - 1:
                    val_loss = self.validate()
                    info = 'epoch: {} - val loss: {}'.format(epoch, val_loss)
                    print(info)
                    self.save_checkpoint(self.checkpoint)
                    torch.save(self.model.state_dict(), self.export_weights)
        
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

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        
        optim = ScheduledOptim(
            Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
            0.2, self.config['transformer']['d_model'], self.config['optimizer']['n_warmup_steps'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']

    def save_checkpoint(self, filename):
        state = {'epoch': self.epoch, 'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'train_losses': self.train_losses}
        
        torch.save(state, filename)


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
