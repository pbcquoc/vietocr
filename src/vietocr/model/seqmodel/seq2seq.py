import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
                
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        """
        src: src_len x batch_size x img_channel
        outputs: src_len x batch_size x hid_dim 
        hidden: batch_size x hid_dim
        """

        embedded = self.dropout(src)
        
        outputs, hidden = self.rnn(embedded)
                                 
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        """
        hidden: batch_size x hid_dim
        encoder_outputs: src_len x batch_size x hid_dim,
        outputs: batch_size x src_len
        """
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
  
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim = 1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        """
        inputs: batch_size
        hidden: batch_size x hid_dim
        encoder_outputs: src_len x batch_size x hid_dim
        """
             
        input = input.unsqueeze(0)
        
        embedded = self.dropout(self.embedding(input))
        
        a = self.attention(hidden, encoder_outputs)
                
        a = a.unsqueeze(1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        weighted = torch.bmm(a, encoder_outputs)
        
        weighted = weighted.permute(1, 0, 2)
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        
        return prediction, hidden.squeeze(0), a.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, encoder_hidden, decoder_hidden, img_channel, decoder_embedded, dropout=0.1):
        super().__init__()
        
        attn = Attention(encoder_hidden, decoder_hidden)
        
        self.encoder = Encoder(img_channel, encoder_hidden, decoder_hidden, dropout)
        self.decoder = Decoder(vocab_size, decoder_embedded, encoder_hidden, decoder_hidden, dropout, attn)
        
    def forward_encoder(self, src):       
        """
        src: timestep x batch_size x channel
        hidden: batch_size x hid_dim
        encoder_outputs: src_len x batch_size x hid_dim
        """

        encoder_outputs, hidden = self.encoder(src)

        return (hidden, encoder_outputs)

    def forward_decoder(self, tgt, memory):
        """
        tgt: timestep x batch_size 
        hidden: batch_size x hid_dim
        encouder: src_len x batch_size x hid_dim
        output: batch_size x 1 x vocab_size
        """
        
        tgt = tgt[-1]
        hidden, encoder_outputs = memory
        output, hidden, _ = self.decoder(tgt, hidden, encoder_outputs)
        output = output.unsqueeze(1)
        
        return output, (hidden, encoder_outputs)

    def forward(self, src, trg):
        """
        src: time_step x batch_size
        trg: time_step x batch_size
        outputs: batch_size x time_step x vocab_size
        """

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        device = src.device

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(device)
        encoder_outputs, hidden = self.encoder(src)
                
        for t in range(trg_len):
            input = trg[t] 
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            
            outputs[t] = output
            
        outputs = outputs.transpose(0, 1).contiguous()

        return outputs

    def expand_memory(self, memory, beam_size):
        hidden, encoder_outputs = memory
        hidden = hidden.repeat(beam_size, 1)
        encoder_outputs = encoder_outputs.repeat(1, beam_size, 1)

        return (hidden, encoder_outputs)
    
    def get_memory(self, memory, i):
        hidden, encoder_outputs = memory
        hidden = hidden[[i]]
        encoder_outputs = encoder_outputs[:, [i],:]

        return (hidden, encoder_outputs)
