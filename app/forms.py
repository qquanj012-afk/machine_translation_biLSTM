# app/forms.py
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
from wtforms.validators import DataRequired, Length


class TranslationForm(FlaskForm):
    """Form nhập câu tiếng Anh để dịch"""
    source_text = StringField(
        'Nhập câu tiếng Anh:',
        validators=[
            DataRequired(message='Vui lòng nhập câu.'),
            Length(min=1, max=200, message='Câu không được quá 200 ký tự.')
        ],
        render_kw={'placeholder': 'Ví dụ: How are you today?', 'rows': 3}
    )

    # Tuỳ chọn: chọn phương pháp giải mã (greedy hoặc beam search)
    decoding_method = SelectField(
        'Phương pháp giải mã:',
        choices=[('greedy', 'Greedy'), ('beam', 'Beam Search (chậm hơn)')],
        default='greedy'
    )

    submit = SubmitField('Dịch')