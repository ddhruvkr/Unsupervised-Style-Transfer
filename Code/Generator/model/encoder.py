import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from config import model_config as config

device = torch.device("cuda:"+str(config['gpu']) if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_weights, embedding_dim, dropout_p):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(input_size, self.embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(self.embedding_dim, hidden_size, num_layers = self.num_layers)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_weights).float(), requires_grad=False)

    def forward(self, input, input_len, hidden):
        embedded = self.dropout(self.embedding(input)).permute(1, 0, 2)
        #print(input_len)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_len, enforce_sorted=False)
        #embedded = self.dropout(embedded)
        packed_outputs, hidden = self.gru(packed_embedded, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        #print(packed_outputs.shape)
        #print(output)
        #print(output.shape)
        return output, hidden

    def initHidden(self, batch_size, evaluate):
        if evaluate:
            return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)
        else:
            return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
