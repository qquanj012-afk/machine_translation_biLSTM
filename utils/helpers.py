# utils/helpers.py
import torch
import numpy as np
from typing import List

def compute_bleu(reference: List[str], hypothesis: List[str]) -> float:
    """
    Tính BLEU score đơn giản (dùng nltk nếu có, fallback thủ công).
    Cài đặt nltk: pip install nltk
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smooth = SmoothingFunction().method1
        return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smooth)
    except ImportError:
        # Fallback: tính độ chính xác unigram
        ref_words = set(reference.split())
        hyp_words = hypothesis.split()
        if not hyp_words:
            return 0.0
        matches = sum(1 for w in hyp_words if w in ref_words)
        return matches / len(hyp_words)

def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Tạo mask (True cho vị trí không phải PAD, False cho PAD)
    shape: (batch, seq_len)
    """
    return (seq != pad_idx)

def create_subsequent_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Tạo mask hình tam giác trên cho decoder (che các token tương lai)
    shape: (seq_len, seq_len)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask

def count_parameters(model: torch.nn.Module) -> int:
    """Đếm số lượng tham số trainable của model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)