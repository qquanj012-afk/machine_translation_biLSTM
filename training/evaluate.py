# training/evaluate.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from typing import List

def evaluate(model: nn.Module, dataloader: DataLoader,
             criterion: nn.Module, device: torch.device) -> float:
    """
    Tính average loss trên dataloader (validation/test).
    Đây là hàm evaluate chính, được gọi từ train.py.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            # Forward: decoder input = tgt bỏ token cuối
            output, _ = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate_bleu(model: nn.Module, dataloader: DataLoader,
                  tokenizer, device: torch.device, max_len: int = 50) -> float:
    """
    Tính BLEU score trên dataloader (dùng greedy decoding).
    """
    model.eval()
    references = []
    hypotheses = []
    smoothing = SmoothingFunction().method1

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            batch_size = src.shape[0]
            translations = []
            for i in range(batch_size):
                src_sentence = src[i:i+1]
                tgt_indices = _greedy_decode(model, src_sentence, tokenizer, max_len, device)
                tgt_text = tokenizer.decode(tgt_indices, remove_special=True)
                translations.append(tgt_text)
            for i in range(batch_size):
                ref_tokens = tokenizer.decode(tgt[i].tolist(), remove_special=True).split()
                references.append([ref_tokens])
                hyp_tokens = translations[i].split()
                hypotheses.append(hyp_tokens)
    bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothing)
    return bleu

def _greedy_decode(model, src: torch.Tensor, tokenizer, max_len: int, device: torch.device) -> List[int]:
    """Greedy decoding, trả về list index (không bao gồm <SOS>)."""
    model.eval()
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src)
        sos_idx = tokenizer.word2idx.get('<SOS>', 1)
        eos_idx = tokenizer.word2idx.get('<EOS>', 2)
        tgt = torch.tensor([[sos_idx]]).to(device)
        decoded = []
        for _ in range(max_len):
            output, hidden, cell = model.decoder(tgt, encoder_outputs, hidden, cell)
            next_token = output.argmax(dim=-1).item()
            if next_token == eos_idx:
                break
            decoded.append(next_token)
            tgt = torch.tensor([[next_token]]).to(device)
    return decoded