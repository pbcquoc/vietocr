from vietocr.optim.optim import ScheduledOptim
from vietocr.optim.labelsmoothingloss import LabelSmoothingLoss
from torch.optim import Adam, SGD
from torch import nn
from vietocr.tool.translate import build_model
from vietocr.tool.translate import translate
from vietocr.tool.utils import download_weights
from vietocr.tool.logger import Logger
import yaml
import torch
#from vietocr.loader.DataLoader import DataGen
from vietocr.loader.dataloader import OCRDataset, ClusterRandomSampler, collate_fn
from torch.utils.data import DataLoader

from vietocr.tool.utils import compute_accuracy
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import time

class Trainer():
    def __init__(self, config, pretrained=True):

        self.config = config
        self.model, self.vocab = build_model(config)

        self.device = config['device']
        self.num_iters = config['trainer']['iters']
        self.data_root = config['trainer']['data_root']
        self.train_annotation = config['trainer']['train_annotation']
        self.valid_annotation = config['trainer']['valid_annotation']
        self.batch_size = config['trainer']['batch_size']
        self.print_every = config['trainer']['print_every']
        self.valid_every = config['trainer']['valid_every']
        self.checkpoint = config['trainer']['checkpoint']
        self.export_weights = config['trainer']['export']
        self.metrics = config['trainer']['metrics']
        logger = config['trainer']['log']
    
        if logger:
            self.logger = Logger(logger) 

        if pretrained:
            download_weights(**config['pretrain'], quiet=config['quiet'])
            self.model.load_state_dict(torch.load(config['pretrain']['cached'], map_location=torch.device(self.device)))

        self.iter = 0

        self.optimizer = ScheduledOptim(
#            Adam(self.model.transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
            SGD(self.model.transformer.parameters(), lr=0.01, momentum=0.9, nesterov=True),
            config['optimizer']['init_lr'], config['transformer']['d_model'], config['optimizer']['n_warmup_steps'])

#        self.criterion = nn.CrossEntropyLoss(ignore_index=0) 
        self.criterion = LabelSmoothingLoss(len(self.vocab), padding_idx=self.vocab.pad, smoothing=0.1)

        self.train_gen = self.data_gen('train_db', self.data_root, self.train_annotation)
        if self.valid_annotation != None:
            self.valid_gen = self.data_gen('valid_db', self.data_root, self.valid_annotation)

        self.train_losses = []
        
    def train(self):
        total_loss = 0
        
        total_loader_time = 0
        total_gpu_time = 0

        data_iter = iter(self.train_gen)
        for i in range(self.num_iters):
            self.iter += 1

            start = time.time()

            try:
                batch = next(data_iter)
            except:
                data_iter = iter(self.train_gen)
                batch = next(data_iter)

            total_loader_time += time.time() - start

            start = time.time()
            loss = self.step(batch)
            total_gpu_time += time.time() - start

            total_loss += loss
            self.train_losses.append((self.iter, loss))

            if self.iter % self.print_every == 0:
                info = 'iter: {:06d} - train loss: {:.4f} - lr: {:.4e} - load time: {:.3f} - gpu time: {:.3f}'.format(self.iter, 
                        total_loss/self.print_every, self.optimizer.lr, 
                        total_loader_time, total_gpu_time)

                total_loss = 0
                total_loader_time = 0
                total_gpu_time = 0
                print(info) 
                self.logger.log(info)

            if self.valid_annotation and self.iter % self.valid_every == 0:
                val_loss = self.validate()
                info = 'iter: {:06d} - val loss: {:.4f}'.format(self.iter, val_loss)
                print(info)
                self.logger.log(info)

                acc_full_seq, acc_per_char = self.precision(self.metrics)
                info = 'iter: {:06d} - acc full seq: {:.4f} - acc per char: {:.4f}'.format(self.iter, acc_full_seq, acc_per_char)
                print(info)
                self.logger.log(info)

                self.save_checkpoint(self.checkpoint)
                self.save_weight(self.export_weights)
        
    def validate(self):
        self.model.eval()

        total_loss = []
        
        with torch.no_grad():
            for step, batch in enumerate(self.valid_gen):
                batch = self.batch_to_device(batch)
                img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch['tgt_input'], batch['tgt_output'], batch['tgt_padding_mask']

                outputs = self.model(img, tgt_input, tgt_padding_mask)
                #loss = self.criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_output, 'b o -> (b o)'))
               
                outputs = outputs.view(-1, outputs.shape[-1])   
                tgt_output = tgt_output.view(-1)

                loss = self.criterion(outputs, tgt_output)

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
        for batch in  self.valid_gen:
            batch = self.batch_to_device(batch)
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

    def precision(self, sample=None):

        pred_sents, actual_sents, _ = self.predict(sample=sample)

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
        self.iter = checkpoint['iter']

        self.train_losses = checkpoint['train_losses']

    def save_checkpoint(self, filename):
        state = {'iter':self.iter, 'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'train_losses': self.train_losses}
        
        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)

        torch.save(state, filename)

    
    def save_weight(self, filename):
        path, _ = os.path.split(filename)
        os.makedirs(path, exist_ok=True)
       
        torch.save(self.model.state_dict(), filename)

    def batch_to_device(self, batch):
        img = batch['img'].to(self.device, non_blocking=True)
        tgt_input = batch['tgt_input'].to(self.device, non_blocking=True)
        tgt_output = batch['tgt_output'].to(self.device, non_blocking=True)
        tgt_padding_mask = batch['tgt_padding_mask'].to(self.device, non_blocking=True)

        batch = {
                'img': img, 'tgt_input':tgt_input, 
                'tgt_output':tgt_output, 'tgt_padding_mask':tgt_padding_mask, 
                'filenames': batch['filenames']
                }

        return batch

    def data_gen(self, lmdb_path, data_root, annotation):
        dataset = OCRDataset(lmdb_path=lmdb_path, root_dir=data_root, annotation_path=annotation, vocab=self.vocab, **self.config['dataset'])
        sampler = ClusterRandomSampler(dataset, self.batch_size, True)
        gen = DataLoader(
                dataset,
                batch_size=self.batch_size, 
                sampler=sampler,
                collate_fn = collate_fn,
                shuffle=False,
                drop_last=False,
                **self.config['dataloader'])
       
        return gen

    def step(self, batch):
        self.model.train()

        batch = self.batch_to_device(batch)
        img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch['tgt_input'], batch['tgt_output'], batch['tgt_padding_mask']    
        
        outputs = self.model(img, tgt_input, tgt_key_padding_mask=tgt_padding_mask)
#        loss = self.criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_output, 'b o -> (b o)'))
        outputs = outputs.view(-1, outputs.shape[-1])   
        tgt_output = tgt_output.view(-1)

        loss = self.criterion(outputs, tgt_output)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step_and_update_lr()
        
        loss_item = loss.item()

        del outputs
        del loss

        return loss_item
