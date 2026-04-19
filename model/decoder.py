# model/decoder.py
import torch
import torch.nn as nn
from model.attention import Attention

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int,
                 num_layers: int = 2, dropout: float = 0.3, attention_type: str = 'general'):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embedding_dim + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.attention = Attention(hidden_size, attention_type)
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, tgt, encoder_outputs, hidden, cell):
        batch_size = tgt.shape[0]
        tgt_len = tgt.shape[1]

        embedded = self.dropout(self.embedding(tgt))

        # Lấy backward hidden states (unidirectional)
        hidden_uni = hidden[-self.num_layers:]   # (num_layers, batch, hidden_size)
        cell_uni = cell[-self.num_layers:]

        outputs = torch.zeros(batch_size, tgt_len, self.fc_out.out_features).to(tgt.device)

        for t in range(tgt_len):
            x = embedded[:, t, :].unsqueeze(1)  # (batch, 1, emb_dim)

            # hidden state của tầng cuối (batch, hidden_size)
            decoder_hidden = hidden_uni[-1]

            # Attention: context (batch, hidden_size)
            context, _ = self.attention(decoder_hidden, encoder_outputs)

            # Thêm chiều thứ 2 để concat với x
            context = context.unsqueeze(1)   # (batch, 1, hidden_size)

            lstm_input = torch.cat((x, context), dim=-1)  # (batch, 1, emb_dim + hidden_size)
            lstm_output, (hidden_uni, cell_uni) = self.lstm(lstm_input, (hidden_uni, cell_uni))

            prediction = self.fc_out(torch.cat((lstm_output, context), dim=-1))
            outputs[:, t, :] = prediction.squeeze(1)

        return outputs, hidden_uni, cell_uni