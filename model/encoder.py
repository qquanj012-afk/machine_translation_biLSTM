# model/encoder.py
import torch
import torch.nn as nn

class Encoder(nn.Module):
    """Bộ mã hóa biLSTM"""
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int,
                 num_layers: int = 2, dropout: float = 0.3, bidirectional: bool = True):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        # LSTM (bidirectional)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Nếu bidirectional, thêm lớp Linear để giảm chiều từ hidden_size*2 -> hidden_size
        if bidirectional:
            self.fc_output = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.fc_output = nn.Identity()

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)  # outputs: (batch, seq_len, hidden_size*num_directions)
        outputs = self.fc_output(outputs)              # (batch, seq_len, hidden_size)
        return outputs, hidden, cell