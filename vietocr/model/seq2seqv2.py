import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, emb_dim, enc_hid_dim, dec_hid_dim, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.enc_hid_dim = enc_hid_dim
        
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, num_layers=num_layers, bidirectional = True)
        
        self.fc = nn.Linear(enc_hid_dim, dec_hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        """
        src: src_len x batch_size x img_channel
        outputs: src_len x batch_size x enc_hid_dim 
        hidden: num_layers x batch_size x dec_hid_dim
        """

        
        batch_size = src.shape[1]
        embedded = self.dropout(src)
        
        #embedded = [src len, batch size, emb dim]
                
                
        outputs, hidden = self.rnn(embedded)
                                 
        #packed_outputs is a packed sequence containing all hidden states
        #hidden is now from the final non-padded element in the batch
            
            
        #outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros
            
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        hidden = hidden.view(self.num_layers, 2, batch_size, self.enc_hid_dim)
        hidden = torch.sum(hidden, dim=1)
        hidden = torch.tanh(self.fc(hidden))
        
        outputs = outputs[:,:,:self.enc_hid_dim] + outputs[:,:, self.enc_hid_dim:]
        

        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        """
        hidden: num_layers x batch_size x dec_hid_dim
        encoder_outputs: src_len x batch_size x enc_hid_dim,
        outputs: batch_size x src_len
        """
        hidden = torch.sum(hidden, dim=0)
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
  
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention = [batch size, src len]
        
#         attention = attention.masked_fill(mask == 0, -1e10)
        
        return F.softmax(attention, dim = 1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, num_layers, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU(enc_hid_dim + emb_dim, dec_hid_dim, num_layers=num_layers)
        
        self.fc_out = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        """
        inputs: batch_size
        hidden: num_layers x batch_size x dec_hid_dim
        encoder_outputs: src_len x batch_size x enc_hid_dim
        """
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        #mask = [batch size, src len]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
        
        a = self.attention(hidden, encoder_outputs)
                
        #a = [batch size, src len]
        
        a = a.unsqueeze(1)
        
        #a = [batch size, 1, src len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        weighted = torch.bmm(a, encoder_outputs)
        
        #weighted = [batch size, 1, enc hid dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        
        #weighted = [1, batch size, enc hid dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
            
        output, hidden = self.rnn(rnn_input, hidden)
        
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
#         assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, a.squeeze(1)

    
class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, encoder_hidden, decoder_hidden, img_channel, decoder_embedded, num_layers, dropout=0.1):
        super().__init__()
        
        attn = Attention(encoder_hidden, decoder_hidden)

        self.encoder = Encoder(img_channel, encoder_hidden, decoder_hidden, num_layers, dropout)
        self.decoder = Decoder(vocab_size, decoder_embedded, encoder_hidden, decoder_hidden, num_layers, dropout, attn)

    def forward_encoder(self, src):       
        """src: timestep, batch size, channel
           hidden: batchsize x dim
           encoder_outputs: src len, batch size, enc hid dim * 2
        """
        encoder_outputs, hidden = self.encoder(src)
        self.encoder_outputs = encoder_outputs

        return hidden

    def forward_decoder(self, tgt, memory):
        """tgt: timestep x batchsize 
           output: batch size x 1 x vocabsize
        """
        tgt = tgt[-1]
        output, hidden, _ = self.decoder(tgt, memory, self.encoder_outputs)
        output = output.unsqueeze(1)
        
        return output, hidden

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        #src = [src len, batch size, channel]
        #src_len = [batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
                    
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        device = src.device

        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)
                
        #first input to the decoder is the <sos> tokens
#        input = trg[0]
        
#         mask = self.create_mask(src)

        #mask = [batch size, src len]
                
        for t in range(trg_len):
            input = trg[t] 
            #insert input token embedding, previous hidden state, all encoder hidden states 
            #  and mask
            #receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
#            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
#            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
#            input = trg[t+1] if t+1 < trg_len and teacher_force else top1
        
        outputs = outputs.transpose(0, 1).contiguous()

        # outputs batch_size, trg_len, vocab_size
        return outputs   

