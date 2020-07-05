from vietocr.model.cnn import CNN
from vietocr.model.transformer import LanguageTransformer
from torch import nn

class VietOCR(nn.Module):
    def __init__(self, vocab_size,
                 backbone,
                 cnn_args, 
                 transformer_args):
        
        super(VietOCR, self).__init__()
        
        self.cnn = CNN(backbone, **cnn_args)
        self.transformer=LanguageTransformer(vocab_size, **transformer_args)

    def forward(self, img, tgt_input, tgt_key_padding_mask):
        """
        Shape:
            - img: (N, C, H, W)
            - tgt_input: (T, N)
            - tgt_key_padding_mask: (N, T)
            - output: b t v
        """
        src = self.cnn(img)
        outputs = self.transformer(src, tgt_input, tgt_key_padding_mask=tgt_key_padding_mask)
        
        return outputs

