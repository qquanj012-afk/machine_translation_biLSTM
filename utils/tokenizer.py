# utils/tokenizer.py
import os
import pickle
from collections import Counter
from typing import List, Tuple, Dict


class Tokenizer:
    def __init__(self, config: dict):
        self.special_tokens = config['tokenizer']['special_tokens']
        self.max_seq_len = config['data']['max_seq_len']
        self.min_freq = config['data']['min_freq']

        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

        # Tạo các token đặc biệt
        for token in self.special_tokens:
            self.add_word(token)

    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1
        return self.word2idx[word]

    def build_vocab(self, sentences: List[str]):
        """Xây dựng từ điển từ list câu (đã tokenize thô)"""
        counter = Counter()
        for sent in sentences:
            # Tokenize đơn giản: tách khoảng trắng
            tokens = sent.lower().split()
            counter.update(tokens)

        # Thêm các từ đủ tần suất
        for word, freq in counter.items():
            if freq >= self.min_freq:
                self.add_word(word)

        # Đảm bảo có token <UNK>
        self.unk_idx = self.word2idx.get('<UNK>', 0)

    def encode(self, sentence: str, add_sos: bool = True, add_eos: bool = True) -> List[int]:
        """Chuyển câu thành list index"""
        tokens = sentence.lower().split()
        if add_sos:
            indices = [self.word2idx.get('<SOS>', self.unk_idx)]
        else:
            indices = []
        for token in tokens:
            idx = self.word2idx.get(token, self.unk_idx)
            indices.append(idx)
        if add_eos:
            indices.append(self.word2idx.get('<EOS>', self.unk_idx))
        # Cắt theo max_seq_len
        if len(indices) > self.max_seq_len:
            indices = indices[:self.max_seq_len]
        return indices

    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """Chuyển list index thành câu"""
        tokens = []
        for idx in indices:
            word = self.idx2word.get(idx, '<UNK>')
            if remove_special and word in self.special_tokens:
                continue
            tokens.append(word)
        return ' '.join(tokens)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'vocab_size': self.vocab_size,
                'special_tokens': self.special_tokens,
                'max_seq_len': self.max_seq_len,
                'min_freq': self.min_freq
            }, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.vocab_size = data['vocab_size']
        self.special_tokens = data['special_tokens']
        self.max_seq_len = data['max_seq_len']
        self.min_freq = data['min_freq']
        self.unk_idx = self.word2idx.get('<UNK>', 0)