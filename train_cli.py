# train_cli.py
import os
import torch
import torch.nn as nn
from torch.optim import Adam

from utils.config_loader import get_config
from utils.tokenizer import Tokenizer
from utils.data_loader import build_dataloaders
from model.encoder import Encoder
from model.decoder import Decoder
from model.seq2seq import Seq2Seq
from training.train import train


def main():
    config = get_config()
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Khởi tạo tokenizer
    tokenizer = Tokenizer(config)

    # Xây dựng dataloader và vocab
    train_loader, val_loader, tokenizer = build_dataloaders(config, tokenizer)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    # Lưu tokenizer đã train
    processed_dir = config['data']['processed_dir']
    os.makedirs(processed_dir, exist_ok=True)
    tokenizer.save(os.path.join(processed_dir, 'tokenizer.pkl'))

    # Các tham số model
    embedding_dim = config['model']['encoder_hidden_size']  # có thể tách riêng, nhưng dùng chung cho đơn giản
    encoder_hidden = config['model']['encoder_hidden_size']
    decoder_hidden = config['model']['decoder_hidden_size']
    num_layers = config['model']['num_layers']
    dropout = config['model']['dropout']
    bidirectional = config['model']['bidirectional']
    attention_type = config['model']['attention_type']

    # Khởi tạo encoder và decoder
    encoder = Encoder(vocab_size, embedding_dim, encoder_hidden, num_layers, dropout, bidirectional)
    decoder = Decoder(vocab_size, embedding_dim, decoder_hidden, num_layers, dropout, attention_type)

    # Seq2Seq model
    model = Seq2Seq(encoder, decoder, device)
    model = model.to(device)

    # Optimizer và loss (bỏ qua ignore index cho padding)
    optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])
    pad_idx = tokenizer.word2idx.get('<PAD>', 0)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Huấn luyện
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=config['training']['epochs'],
        checkpoint_dir=config['training']['checkpoint_dir'],
        log_file=os.path.join(config['training']['log_dir'], 'train_log.csv'),
        clip=config['training']['clip'],
        patience=config['training']['patience']
    )

    print("Training completed!")


if __name__ == "__main__":
    main()