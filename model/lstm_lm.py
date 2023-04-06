'''Defines the neural network, loss function, and metrics'''

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMLanguageModel(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 num_layers,
                 dropout,
                 embeds,
                 pad_idx=1,
                 freeze_embeds=True
                 ):

        super(LSTMLanguageModel, self).__init__()

        self.dummy_param = nn.Parameter(torch.empty(0)) # to get this model's device

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pad_idx = pad_idx

        if embeds is not None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1).from_pretrained(embeds, freeze=freeze_embeds)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True, # first dimension is batch size
                            bidirectional=True)

        self.hidden2out = nn.Linear(hidden_dim * 2, vocab_size)

        self.dropout_layer = nn.Dropout(p=dropout)

    def init_hidden(self, batch_size):
        device = self.dummy_param.device
        return(autograd.Variable(torch.randn(self.num_layers * 2, batch_size, self.hidden_dim)).to(device),
               autograd.Variable(torch.randn(self.num_layers * 2, batch_size, self.hidden_dim)).to(device))

    #def forward(self, text, lens):
    def forward(self, text):
        self.hidden = self.init_hidden(text.shape[0]) # initialized randomly with each forward pass
        embeds = self.embedding(text.long())

        lens = torch.tensor([y.item() for y in map(lambda x: sum(x != self.pad_idx), text)],
                            dtype=torch.int64)

        embeds = pack_padded_sequence(embeds, lens.cpu(), batch_first=True)
        lstm_out, (hn, cn) = self.lstm(embeds, self.hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        output = self.dropout_layer(lstm_out)
        output = self.hidden2out(output)

        return output


def loss_fn(outputs, labels, weight=None):
    # CrossEntropyLoss expects (batch_size, num_classes, hidden_dim)
    #return nn.CrossEntropyLoss()(outputs.permute(0, 2, 1), labels)
    pad_idx = 1
    return nn.CrossEntropyLoss(ignore_index=pad_idx)(outputs.permute(0, 2, 1), labels)


def accuracy(outputs, labels):
    # Divide by batch size and sequence length to get accuracy [0-1] per batch
    #return (outputs.argmax(2) == labels).sum().item() / labels.shape[0] / labels.shape[1]
    # reshape so that each row contains one token
    # (batch_size, seq_len, output_size) -> (batch_size*seq_len, output_size)
    outputs = outputs.view(-1, outputs.shape[2])

    # get predicted class for each token
    outputs = torch.argmax(outputs, axis=1)

    # reshape labels to give flat vector
    # (batch_size, seq_len) -> (batch_size*seq_len,)
    labels = labels.view(-1)

    # generate mask to exclude padding token (has label 1)
    pad_idx = 1
    mask = (labels != pad_idx)

    # compare outputs with labels and divide by number of tokens
    return torch.sum(outputs == labels) / float(torch.sum(mask))

metrics = {
    'accuracy': accuracy
}
