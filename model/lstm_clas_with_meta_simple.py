'''Defines the neural network, loss function, and metrics'''

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 freeze_embeds,
                 lstm_hidden_dim,
                 lstm_num_layers,
                 meta_size,
                 fc_hidden_dim,
                 dropout,
                 output_size,
                 embeds,
                 pad_idx=1
                 ):

        super(LSTMClassifier, self).__init__()

        self.dummy_param = nn.Parameter(torch.empty(0)) # to get this model's device

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.freeze_embeds = freeze_embeds
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.fc_hidden_dim = fc_hidden_dim
        self.meta_size = meta_size
        self.dropout = dropout
        self.pad_idx = pad_idx

        if embeds is not None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx).from_pretrained(embeds, freeze=freeze_embeds)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=lstm_hidden_dim,
                            num_layers=lstm_num_layers,
                            batch_first=True,
                            bidirectional=True)

        self.dropout_layer = nn.Dropout(p=dropout)

        self.fc = nn.Linear(lstm_hidden_dim, fc_hidden_dim)

        self.hidden2out = nn.Linear(meta_size + fc_hidden_dim, output_size)

    def init_hidden(self, batch_size):
        device = self.dummy_param.device
        return(autograd.Variable(torch.randn(self.num_layers * 2, batch_size, self.hidden_dim)).to(device),
               autograd.Variable(torch.randn(self.num_layers * 2, batch_size, self.hidden_dim)).to(device))

    def forward(self, inp):
        meta, text = inp
        # text: # (batch_size, # num_tokens)
        #self.hidden = self.init_hidden(text.shape[0])
        embeds = self.embedding(text.long()) # (batch_size, embed_dim)

        lens = torch.tensor([y.item() for y in map(lambda x: sum(x != self.pad_idx), text)],
                            dtype=torch.int64)
        embeds = pack_padded_sequence(embeds, lens.cpu(), batch_first=True)

        #lstm_out, (hn, cn) = self.lstm(embeds, self.hidden)
        lstm_out, (hn, cn) = self.lstm(embeds)
        # lstm_out: (batch_size, num_tokens, hidden_dim*2), 2 since bidirectional

        #lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # hn is the last hidden state of the sequences
        # hn: (self.num_layers*2 x batch_size x hidden_dim)
        # hn[-1]: (batch_size x hidden_dim)
        output = self.dropout_layer(hn[-1]) # (batch_size, hidden_dim)

        output = self.fc(output)

        # Concatenate metadata
        output = torch.cat([meta, output], axis=1)

        output = self.hidden2out(output) # (batch_size, output_size)

        return output


def loss_fn(outputs, labels, weight=None):
    return nn.CrossEntropyLoss(weight=weight)(outputs, labels)


def accuracy(outputs, labels):
    return (outputs.argmax(1) == labels).sum().item() / labels.shape[0]


metrics = {
    'accuracy': accuracy
}
