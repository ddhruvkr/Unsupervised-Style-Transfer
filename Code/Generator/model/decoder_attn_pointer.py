import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, num_layers, embedding_weights, embedding_dim, dropout_p):
        super(PAttnDecoderRNN, self).__init__()
        self.batch_size=1
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.output_size, self.embedding_dim)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size + self.embedding_dim, self.embedding_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, num_layers = self.num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.Wh = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.Uh = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_weights).float(), requires_grad=True)
        #self.bg = nn.Parameter(self.batch_size, 1)
        self.W_g = nn.Linear(self.hidden_size, 1, bias=False)
        self.V_g = nn.Linear(self.embedding_dim, 1, bias=False)
        self.U_g = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, input, hidden, encoder_outputs, mask, input_tensor):


        embedded = self.embedding(input).permute(1, 0, 2)
        embedded = self.dropout(embedded)
        output, hidden = self.gru(embedded, hidden)
        # hidden = s_t
        # embedded = d_t
        #print(encoder_outputs.shape) #max_length, batch_size, dim
        #print(hidden[-1].shape) #batch_size, dim, 1
        encoder_outputs_s = encoder_outputs.transpose(0,1) #batch_size, max_len, dim
        #print(encoder_outputs.shape)
        #print(hidden.shape)
        #print(hidden[-1].shape)
        score = torch.bmm(encoder_outputs_s, hidden[-1].unsqueeze(2)).squeeze(2)
        #print(score.shape) #bts, max_len
        #print(score)
        #print(mask)
        score = score.masked_fill(mask == 0, -1e10)
        #print(score.shape)
        #print(score)
        attn_weights = F.softmax(score, dim=1)
        attn_weights = attn_weights.unsqueeze(2)
        #(bts, dim, max_len)(bts, max_len, 1)
        c_t = torch.bmm(encoder_outputs_s.transpose(1,2),attn_weights)
        s_t = hidden[-1]
        print(hidden[-1].shape)
        print(embedded.squeeze(1).shape)
        print(c_t.squeeze(2).shape)
        d_t = embedded.squeeze(1)
        a = self.V_g(d_t)
        print(a.shape)
        b = self.U_g(s_t)
        print(b.shape)
        c = self.W_g(c_t.squeeze(2))
        print(c.shape)
        p_g = torch.sigmoid(a + b + c)
        print(p_g.shape)
        #print(embedded.shape) #bts, 1, dim
        #print(attn_applied.shape) #bts, dim, 1
        W_hc_t = self.Wh(c_t.squeeze(2))
        #print(W_hc_t.shape)
        #print(hidden.shape)
        #print(hidden[-1].shape)
        U_hh_t = self.Uh(hidden[-1])
        g = self.out(torch.tanh(U_hh_t + W_hc_t)).unsqueeze(0)
        print(g.shape)
        print(attn_weights.shape)
        g = (1 - p_g)*attn_weights + p_g*g
        g = F.log_softmax(g, dim=2)
        print(g.shape)
        print(input_tensor)
        print(input_tensor.shape)
        return g, hidden, attn_weights







        '''c_t = attn_applied.squeeze()
        s_t = hidden[-1].squeeze()
        d_t = input #or embedded
        
        σ(W_gc_t + U_gs_t + V_gd_t + bg)
        bg - > batch_size
        #vd = self.V_g(d_t)
        #us = self.U_g(s_t)
        #wc = self.W_g(c_t)
        p_g = torch.softmax(self.V_g(d_t) + self.U_g(s_t) + self.W_g(c_t) + bg)
        #(batch_size, 1)
        output, hidden = self.gru(output.transpose(0,1), hidden)



        output = self.out(output)
        output = F.log_softmax(output, dim=2)
        return output, hidden, attn_weights'''

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
