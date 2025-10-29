import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class EditSequenceDataset(Dataset):
    def __init__(self, csv_file, vocab, max_len):
        self.sequences_df = pd.read_csv(csv_file)
        self.sequences_df.dropna(subset=['sequence'], inplace=True)
        self.vocab = vocab
        self.max_len = max_len
        
        self.sequences_as_int = []
        for seq in self.sequences_df['sequence']:
            self.sequences_as_int.append([self.vocab.get(token, 0) for token in seq.split(',')])

    def __len__(self):
        return len(self.sequences_as_int)

    def __getitem__(self, idx):
        seq = self.sequences_as_int[idx]
        
        input_seq = seq[:-1]
        target_seq = seq[1:]
        
        # Pad the sequences
        padded_input = np.array(input_seq[:self.max_len] + [self.vocab['<pad>']]*(self.max_len - len(input_seq)) if len(input_seq) < self.max_len else input_seq[:self.max_len])
        padded_target = np.array(target_seq[:self.max_len] + [self.vocab['<pad>']]*(self.max_len - len(target_seq)) if len(target_seq) < self.max_len else target_seq[:self.max_len])

        return torch.from_numpy(padded_input), torch.from_numpy(padded_target)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, nhid, nlayers, dropout=0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.fc = nn.Linear(embedding_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.fc(output)
        return output
