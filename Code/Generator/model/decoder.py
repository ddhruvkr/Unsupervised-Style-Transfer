import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from config import model_config as config

device = torch.device("cuda:"+str(config['gpu']) if torch.cuda.is_available() else "cpu")

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, num_layers, embedding_weights, embedding_dim, dropout_p):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.output_size, self.embedding_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, num_layers = self.num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_weights).float(), requires_grad=False)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_outputs, mask):
        embedded = self.embedding(input).permute(1, 0, 2)
        embedded = self.dropout(embedded)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output[0]))
        #print(output.shape)
        #cba = F.gumbel_softmax(self.out(output[0]), tau=0.9, hard=True, eps=1e-10, dim=1)
        return 0, output.unsqueeze(0), hidden, embedded

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
