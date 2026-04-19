# utils/data_loader.py
import os
import torch
from torch.utils.data import DataLoader, random_split
from training.dataset import TranslationDataset
from utils.tokenizer import Tokenizer


def load_paraphrase_data(file_path: str):
    """
    Đọc file paraphrase (mỗi dòng: câu1 \t câu2)
    Trả về list các tuple (src_sentence, tgt_sentence)
    """
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                src = parts[0].strip()
                tgt = parts[1].strip()
                pairs.append((src, tgt))
    return pairs


def build_dataloaders(config, tokenizer: Tokenizer):
    """
    Đọc dữ liệu raw, tokenize, tạo DataLoader train/val
    """
    raw_train_path = config['data']['raw_train_path']
    raw_test_path = config['data']['raw_test_path']
    batch_size = config['training']['batch_size']
    max_seq_len = config['data']['max_seq_len']
    train_ratio = config['data']['train_ratio']
    seed = config['data']['seed']

    # Đọc tất cả cặp từ train và test (gộp lại để chia train/val sau)
    train_pairs = load_paraphrase_data(raw_train_path)
    test_pairs = load_paraphrase_data(raw_test_path)
    all_pairs = train_pairs + test_pairs

    # Xây dựng vocab từ tất cả câu nguồn và đích
    all_src_sentences = [src for src, _ in all_pairs]
    all_tgt_sentences = [tgt for _, tgt in all_pairs]
    tokenizer.build_vocab(all_src_sentences + all_tgt_sentences)

    # Mã hóa tất cả cặp
    src_sequences = []
    tgt_sequences = []
    for src, tgt in all_pairs:
        src_enc = tokenizer.encode(src, add_sos=True, add_eos=True)
        tgt_enc = tokenizer.encode(tgt, add_sos=True, add_eos=True)
        src_sequences.append(src_enc)
        tgt_sequences.append(tgt_enc)

    # Tạo Dataset
    pad_idx = tokenizer.word2idx.get('<PAD>', 0)
    dataset = TranslationDataset(src_sequences, tgt_sequences, max_seq_len, pad_idx)

    # Chia train/val
    total = len(dataset)
    train_size = int(train_ratio * total)
    val_size = total - train_size
    torch.manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tokenizer