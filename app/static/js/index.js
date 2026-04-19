// index.js - xử lý giao diện trang chủ
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('translate-form');
    const sourceInput = document.getElementById('source_text');
    const methodSelect = document.getElementById('decoding_method');
    const translateBtn = document.getElementById('translate-btn');
    const resultArea = document.getElementById('result-area');
    const translationDiv = document.getElementById('translation-text');
    const loadingDiv = document.getElementById('loading');
    const attentionViz = document.getElementById('attention-viz');

    function showError(message) {
        alert('Lỗi: ' + message);
    }

    function escapeHtml(str) {
        return str.replace(/[&<>]/g, function(m) {
            if (m === '&') return '&amp;';
            if (m === '<') return '&lt;';
            if (m === '>') return '&gt;';
            return m;
        });
    }

    function renderAttentionHeatmap(attentionWeights, sourceTokens, targetTokens) {
        if (!attentionWeights || attentionWeights.length === 0) {
            attentionViz.innerHTML = '<p><em>Không có attention weights để hiển thị.</em></p>';
            return;
        }
        let html = '<h4>Attention Weights (heatmap):</h4><table class="attention-table" border="1" cellpadding="5" style="border-collapse: collapse;">';
        html += '<thead><tr><th>Decoder/Encoder</th>';
        for (let i = 0; i < sourceTokens.length; i++) {
            html += '<th>' + escapeHtml(sourceTokens[i]) + '</th>';
        }
        html += '</tr></thead><tbody>';
        for (let t = 0; t < attentionWeights.length; t++) {
            let targetWord = targetTokens[t] || 'step' + t;
            html += '<tr><td><strong>' + escapeHtml(targetWord) + '</strong></td>';
            for (let s = 0; s < attentionWeights[t].length; s++) {
                let weight = attentionWeights[t][s];
                let intensity = Math.floor(weight * 255);
                let color = 'rgb(255, ' + (255 - intensity) + ', ' + (255 - intensity) + ')';
                html += '<td style="background-color: ' + color + ';">' + weight.toFixed(2) + '</td>';
            }
            html += '</tr>';
        }
        html += '</tbody></table>';
        attentionViz.innerHTML = html;
    }

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        const sourceText = sourceInput.value.trim();
        if (!sourceText) {
            showError('Vui lòng nhập câu tiếng Anh');
            return;
        }
        const decodingMethod = methodSelect.value;

        loadingDiv.style.display = 'block';
        resultArea.style.display = 'none';
        translateBtn.disabled = true;

        try {
            const response = await fetch('/translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source: sourceText, method: decodingMethod })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Lỗi không xác định');

            translationDiv.textContent = data.translation;
            resultArea.style.display = 'block';

            if (data.attention_weights && data.source_tokens && data.target_tokens) {
                renderAttentionHeatmap(data.attention_weights, data.source_tokens, data.target_tokens);
            } else {
                attentionViz.innerHTML = '';
            }
        } catch (error) {
            showError(error.message);
        } finally {
            loadingDiv.style.display = 'none';
            translateBtn.disabled = false;
        }
    });
});