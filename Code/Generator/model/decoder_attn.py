import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from config import model_config as config

device = torch.device("cuda:"+str(config['gpu']) if torch.cuda.is_available() else "cpu")

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, num_layers, embedding_weights, embedding_dim, dropout_p):
        super(AttnDecoderRNN, self).__init__()
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
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_weights).float(), requires_grad=False)

    def forward(self, input, hidden, encoder_outputs, mask):

        embedded = self.embedding(input).permute(1, 0, 2)
        embedded = self.dropout(embedded)
        output, hidden = self.gru(embedded, hidden)
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
        #print(embedded.shape) #bts, 1, dim
        #print(attn_applied.shape) #bts, dim, 1
        W_hc_t = self.Wh(c_t.squeeze(2))
        #print(W_hc_t.shape)
        #print(hidden.shape)
        #print(hidden[-1].shape)
        U_hh_t = self.Uh(hidden[-1])
        g = self.out(torch.tanh(U_hh_t + W_hc_t)).unsqueeze(0)
        #print(g.shape)
        g_ls = F.log_softmax(g, dim=2)
        #print(g_ls)
        #print(g_ls.shape)
        abc = F.gumbel_softmax(g, tau=0.9, hard=False, eps=1e-10, dim=2)
        '''print(abc)
        print(abc.shape)
        abc = F.gumbel_softmax(g_ls, tau=0.9, hard=False, eps=1e-10, dim=2)
        print(abc)
        print(abc.shape)'''

        cba = F.gumbel_softmax(g, tau=0.9, hard=True, eps=1e-10, dim=2)
        #print(cba)
        #print(cba.shape)
        #print(1/0)
        #g = g.squeeze(0)
        return cba, g_ls, hidden, attn_weights



        '''embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        #print(encoder_outputs.shape) #max_length, batch_size, dim
        #print(hidden[-1].shape) #batch_size, dim, 1
        encoder_outputs_s = encoder_outputs.transpose(0,1) #batch_size, max_len, dim

        score = torch.bmm(encoder_outputs_s, hidden[-1].unsqueeze(2)).squeeze(2)
        #print(score.shape) #bts, max_len, 1i
        score = score.masked_fill(mask == 0, -1e10)
        attn_weights = F.softmax(score, dim=1)
        attn_weights = attn_weights.unsqueeze(2)
        #(bts, dim, max_len)(bts, max_len, 1)
        attn_applied = torch.bmm(encoder_outputs_s.transpose(1,2),attn_weights)
        #print(embedded.shape) #bts, 1, dim
        #print(attn_applied.shape) #bts, dim, 1
        output = torch.cat((embedded, attn_applied.transpose(1,2)), 2)
        #print(output.shape) #bts, 1, 2*dim
        output = self.attn_combine(output)
        output = F.relu(output)
        output, hidden = self.gru(output.transpose(0,1), hidden)
        #print(output.shape)
        output = self.out(output)
        #print(output.shape)
        output = F.log_softmax(output, dim=2)
        return output, hidden, attn_weights'''

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
