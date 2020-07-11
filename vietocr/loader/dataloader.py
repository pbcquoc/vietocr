import os
import random

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from vietocr.tool.translate import process_image

class OCRDataset(Dataset):
    def __init__(self, root_dir, annotation_path, vocab, image_height=32, image_min_width=32, image_max_width=512, transform=None):
        self.root_dir = root_dir
        self.annotation_path = os.path.join(root_dir, annotation_path)
        self.vocab = vocab

        self.image_height = image_height
        self.image_min_width = image_min_width
        self.image_max_width = image_max_width
        

        with open(self.annotation_path, 'r') as ann_file:
            lines = ann_file.readlines()
            self.annotations = [l.strip().split('\t') for l in lines]
            
        self.build_cluster_indices()

    def build_cluster_indices(self):
        self.cluster_indices = []
        
        print('build cluster indices')

        for i in range(self.__len__()):
            sample = self.__getitem__(i)
            img = sample['img']
            width = img.shape[-1]

            self.cluster_indices.append(width)
        

    def read_data(self, img_path, lex):

        with open(img_path, 'rb') as img_file:
            img = Image.open(img_file).convert('RGB')
            img_bw = process_image(img, self.image_height, self.image_min_width, self.image_max_width)

        word = self.vocab.encode(lex)

        return img_bw, word

    def __getitem__(self, idx):
        img_path, lex =  self.annotations[idx]
        img_path = os.path.join(self.root_dir, img_path)
        
        img, word = self.read_data(img_path, lex)

        sample = {'img': img, 'word': word, 'img_path': img_path}

        return sample

    def __len__(self):
        return len(self.annotations)

class ClusterRandomSampler(Sampler):
    
    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle        

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):
        batch_lists = []
        for cluster_indices in self.data_source.cluster_indices:
            batches = [cluster_indices[i:i + self.batch_size] for i in range(0, len(cluster_indices), self.batch_size)]
            batches = [_ for _ in batches if len(_) == self.batch_size]
            if self.shuffle:
                random.shuffle(batches)

            batch_lists.append(batches)

        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)

        lst = self.flatten_list(lst)

        return iter(lst)

    def __len__(self):
        return len(self.data_source)
