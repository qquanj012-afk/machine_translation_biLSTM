# model/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size: int, method: str = 'general'):
        super().__init__()
        self.method = method
        self.hidden_size = hidden_size

        if method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif method == 'concat':
            self.attn = nn.Linear(2 * hidden_size, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor = None):
        """
        decoder_hidden: (batch, hidden_size)
        encoder_outputs: (batch, seq_len, hidden_size)
        mask: (batch, seq_len) – True cho vị trí không phải PAD
        return: context vector (batch, hidden_size), attention weights (batch, seq_len)
        """
        batch_size, seq_len, _ = encoder_outputs.shape

        if self.method == 'dot':
            # decoder_hidden (batch, hidden) -> (batch, hidden, 1)
            # scores = encoder_outputs @ decoder_hidden.unsqueeze(2) -> (batch, seq_len, 1)
            scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        elif self.method == 'general':
            # W * decoder_hidden
            w_hidden = self.attn(decoder_hidden)  # (batch, hidden)
            scores = torch.bmm(encoder_outputs, w_hidden.unsqueeze(2)).squeeze(2)
        elif self.method == 'concat':
            # tanh(W [encoder_h; decoder_h])
            decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(-1, seq_len, -1)
            concat = torch.cat((encoder_outputs, decoder_hidden_expanded), dim=2)
            energy = torch.tanh(self.attn(concat))
            scores = torch.bmm(energy, self.v.unsqueeze(1)).squeeze(2)
        else:
            raise ValueError(f"Unknown attention method: {self.method}")

        # Mask padding (gán -inf cho các vị trí PAD)
        if mask is not None:
            scores = scores.masked_fill(mask == False, float('-inf'))

        attn_weights = F.softmax(scores, dim=1)  # (batch, seq_len)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, hidden)
        return context, attn_weights