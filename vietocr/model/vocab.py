class Vocab():
    def __init__(self, chars):
        self.pad = 0
        self.go = 1
        self.eos = 2
        self.chars = chars

        self.c2i = {c:i+3 for i, c in enumerate(chars)}
        self.i2c = {i+3:c for i, c in enumerate(chars)}
        
    def encode(self, chars):
        return [self.go] + [self.c2i[c] for c in chars] + [self.eos]
    
    def decode(self, ids):
        last = ids.index(self.eos) if self.eos in ids else None
        return ''.join([self.i2c[i] for i in ids[1:last]])
    
    def __len__(self):
        return len(self.c2i) + 3
    
    def batch_decode(self, arr):
        texts = [self.decode(ids) for ids in arr]
        return texts

    def __str__(self):
        return self.chars
