# app/__init__.py
import os
import torch
from flask import Flask
from utils.config_loader import get_config
from utils.tokenizer import Tokenizer
from model.encoder import Encoder
from model.decoder import Decoder
from model.seq2seq import Seq2Seq

# Global objects để dùng trong routes
model = None
tokenizer = None
device = None
config = None


def create_app():
    global model, tokenizer, device, config

    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'

    # Load config
    config = get_config()

    # Setup device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')

    # Load tokenizer
    tokenizer_path = os.path.join(config['data']['processed_dir'], 'tokenizer.pkl')
    tokenizer = Tokenizer(config)
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
        print(f"Loaded tokenizer from {tokenizer_path}, vocab_size={tokenizer.vocab_size}")
    else:
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Please run train_cli.py first.")

    # Load model
    vocab_size = tokenizer.vocab_size
    embedding_dim = 256
    encoder = Encoder(vocab_size, embedding_dim,
                      config['model']['encoder_hidden_size'],
                      config['model']['num_layers'],
                      config['model']['dropout'],
                      bidirectional=config['model']['bidirectional'])
    decoder = Decoder(vocab_size, embedding_dim,
                      config['model']['decoder_hidden_size'],
                      config['model']['num_layers'],
                      config['model']['dropout'],
                      attention_type=config['model']['attention_type'])
    seq2seq = Seq2Seq(encoder, decoder, device)

    checkpoint_path = os.path.join(config['training']['checkpoint_dir'], 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        seq2seq.load_state_dict(checkpoint['model_state_dict'])
        seq2seq.to(device)
        print(f"Loaded model from {checkpoint_path}, epoch={checkpoint.get('epoch', '?')}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please train the model first.")

    seq2seq.eval()
    model = seq2seq

    # Register blueprint (routes)
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app