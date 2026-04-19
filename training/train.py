# training/train.py
import os
import csv
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from training.evaluate import evaluate

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device, clip: float = 1.0) -> float:
    """Huấn luyện một epoch, trả về loss trung bình."""
    model.train()
    total_loss = 0.0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output, _ = model(src, tgt[:, :-1])
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
          optimizer: torch.optim.Optimizer, criterion: nn.Module,
          device: torch.device, epochs: int, checkpoint_dir: str = "checkpoints",
          log_file: str = "logs/train_log.csv", clip: float = 1.0, patience: int = 5) -> None:
    """Vòng lặp huấn luyện chính, lưu checkpoint và log CSV, early stopping."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0

    with open(log_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'time_sec'])

        for epoch in range(1, epochs + 1):
            start_time = datetime.now()
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, clip)
            val_loss = evaluate(model, val_loader, criterion, device)   # ok
            elapsed = (datetime.now() - start_time).total_seconds()

            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.2f}s")
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{elapsed:.2f}"])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"  -> Saved best model to {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break