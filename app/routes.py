# app/routes.py
from flask import Blueprint, render_template, request, jsonify
from app import model, tokenizer, device, config
from app.forms import TranslationForm
import torch

main_bp = Blueprint('main', __name__)


@main_bp.route('/', methods=['GET'])
def index():
    form = TranslationForm()
    return render_template('index.html', form=form)


@main_bp.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()
        source_text = data.get('source', '').strip()
        method = data.get('method', 'greedy')
        if not source_text:
            return jsonify({'error': 'Vui lòng nhập câu'}), 400

        # Tokenize câu nguồn
        src_indices = tokenizer.encode(source_text, add_sos=True, add_eos=True)
        src_tensor = torch.tensor([src_indices]).to(device)

        # Dịch
        with torch.no_grad():
            if method == 'greedy':
                tgt_indices = greedy_decode(model, src_tensor, tokenizer, max_len=50)
            else:
                # Beam search (chưa implement, fallback sang greedy)
                tgt_indices = greedy_decode(model, src_tensor, tokenizer, max_len=50)

        # Decode kết quả
        translation = tokenizer.decode(tgt_indices, remove_special=True)

        # Lấy attention weights nếu có (tùy chọn, cần sửa model trả về)
        # Ở đây tạm thời bỏ qua
        return jsonify({
            'translation': translation,
            'source_tokens': source_text.split(),  # đơn giản
            'target_tokens': translation.split(),
            'attention_weights': None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def greedy_decode(model, src_tensor, tokenizer, max_len=50):
    model.eval()
    encoder_outputs, hidden, cell = model.encoder(src_tensor)
    sos_idx = tokenizer.word2idx.get('<SOS>', 1)
    eos_idx = tokenizer.word2idx.get('<EOS>', 2)
    tgt = torch.tensor([[sos_idx]]).to(src_tensor.device)
    decoded = []
    for _ in range(max_len):
        output, hidden, cell = model.decoder(tgt, encoder_outputs, hidden, cell)
        next_token = output.argmax(dim=-1).item()
        if next_token == eos_idx:
            break
        decoded.append(next_token)
        tgt = torch.tensor([[next_token]]).to(src_tensor.device)
    return decoded