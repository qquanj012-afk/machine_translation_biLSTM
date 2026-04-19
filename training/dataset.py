# training/dataset.py
import torch
from torch.utils.data import Dataset
from typing import List, Tuple

class TranslationDataset(Dataset):
    """Dataset cho bài toán dịch máy (sequence-to-sequence)."""
    def __init__(self, source_data: List[List[int]], target_data: List[List[int]],
                 max_seq_len: int = 50, pad_idx: int = 0):
        """
        Args:
            source_data: List các chuỗi index (câu nguồn)
            target_data: List các chuỗi index (câu đích)
            max_seq_len: Độ dài tối đa của câu (cắt hoặc padding)
            pad_idx: index của token <PAD>
        """
        self.source_data = source_data
        self.target_data = target_data
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx

    def __len__(self) -> int:
        return len(self.source_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src = self.source_data[idx][:self.max_seq_len]  # cắt nếu dài hơn
        tgt = self.target_data[idx][:self.max_seq_len]

        # Padding nếu ngắn hơn max_seq_len
        src = src + [self.pad_idx] * (self.max_seq_len - len(src))
        tgt = tgt + [self.pad_idx] * (self.max_seq_len - len(tgt))

        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)