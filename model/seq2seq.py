# model/seq2seq.py
import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder

class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 0.5):
        """
        Huấn luyện (teacher forcing)
        src: (batch, src_len)
        tgt: (batch, tgt_len) - bao gồm <SOS> ở đầu, <EOS> ở cuối
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)

        decoder_input = tgt[:, 0].unsqueeze(1)

        for t in range(1, tgt_len):
            decoder_output, hidden, cell = self.decoder(decoder_input, encoder_outputs, hidden, cell)
            outputs[:, t, :] = decoder_output.squeeze(1)

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            if teacher_force:
                decoder_input = tgt[:, t].unsqueeze(1)
            else:
                # Lấy token có xác suất cao nhất
                top1 = decoder_output.argmax(dim=-1)
                decoder_input = top1

        return outputs, None

    def predict(self, src: torch.Tensor, max_len: int, sos_idx: int, eos_idx: int):
        """
        Suy luận (greedy decoding)
        src: (1, src_len) - một câu nguồn
        """
        self.eval()
        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encoder(src)
            decoder_input = torch.tensor([[sos_idx]]).to(self.device)
            outputs = []
            for _ in range(max_len):
                decoder_output, hidden, cell = self.decoder(decoder_input, encoder_outputs, hidden, cell)
                top1 = decoder_output.argmax(dim=-1).item()
                outputs.append(top1)
                if top1 == eos_idx:
                    break
                decoder_input = torch.tensor([[top1]]).to(self.device)
        return outputs