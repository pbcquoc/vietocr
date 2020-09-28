import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=num_layers, bidirectional = True)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        """
        src: src_len x batch_size x img_channel
        outputs: src_len x batch_size x hid_dim 
        hidden: num_layers x batch_size x hid_dim
        """

        batch_size = src.shape[1]
        embedded = self.dropout(src)
        
        outputs, hidden = self.rnn(embedded)
                                 
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hid_dim)
        hidden = torch.sum(hidden, dim=1)
        
        outputs = outputs[:,:,:self.hid_dim] + outputs[:,:, self.hid_dim:]

        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        
        self.attn = nn.Linear(2*hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        """
        hidden: num_layers x batch_size x hid_dim
        encoder_outputs: src_len x batch_size x hid_dim,
        outputs: batch_size x src_len
        """

        hidden = torch.sum(hidden, dim=0)
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
  
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        energy = F.relu(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim = 1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, num_layers, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU(hid_dim + emb_dim, hid_dim, num_layers=num_layers)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
       
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        """
        inputs: batch_size
        hidden: num_layers x batch_size x hid_dim
        encoder_outputs: src_len x batch_size x hid_dim
        """

        input = input.unsqueeze(0)
        
        embedded = self.dropout(self.embedding(input))
        
        a = self.attention(hidden, encoder_outputs)
                
        a = a.unsqueeze(1) # Bx1xT
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2) # BxTxH
        
        weighted = torch.bmm(a, encoder_outputs) 
        
        weighted = weighted.permute(1, 0, 2)
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        
        output, hidden = self.rnn(rnn_input, hidden)
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(output)
        
        return prediction, hidden , a.squeeze(1)

    
class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, img_dim, hidden, decoder_embedded, num_layers, dropout=0.1):
        super().__init__()
        
        self.vocab_size = vocab_size

        attn = Attention(hidden)

        self.encoder = Encoder(img_dim, hidden, num_layers, dropout)
        self.decoder = Decoder(vocab_size, decoder_embedded, hidden, num_layers, dropout, attn)

    def forward_encoder(self, src):       
        """
        src: timestep x batch_size x channel
        hidden: num_layers x batch_size x hid_dim
        encoder_outputs: src_len x batch_size x hid_dim
        """

        encoder_outputs, hidden = self.encoder(src)

        return (hidden, encoder_outputs)

    def forward_decoder(self, tgt, memory):
        """
        tgt: timestep x batch_size 
        hidden: num_layers x batch_size x hid_dim
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
        device = src.device

        outputs = torch.zeros(trg_len, batch_size, self.vocab_size).to(device)
        
        encoder_outputs, hidden = self.encoder(src)
                
        for t in range(trg_len):
            input = trg[t] 

            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            
            outputs[t] = output
            
        outputs = outputs.transpose(0, 1).contiguous()

        return outputs 
