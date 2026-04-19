/**
 * result.js
 * Xử lý các tương tác trên trang kết quả dịch.
 * Trang này hiển thị câu gốc và câu đã dịch, cho phép quay lại trang chính.
 */

// Đợi DOM tải xong mới thực thi
document.addEventListener('DOMContentLoaded', function() {
    // Lấy các phần tử cần thao tác (nếu có)
    const backLink = document.querySelector('.back-link');
    const sourceText = document.querySelector('.source-area p');
    const targetText = document.querySelector('.target-area p');

    // In ra console để kiểm tra (có thể xoá sau khi chạy ổn định)
    console.log('Trang kết quả đã sẵn sàng.');

    // Nếu có thông báo hoặc analytics, có thể thêm ở đây
    if (sourceText && targetText) {
        console.log('Câu gốc:', sourceText.innerText);
        console.log('Câu dịch:', targetText.innerText);
    }

    // Ví dụ: Thêm sự kiện cho backLink (tuy nhiên thẻ <a> đã có href, không cần thiết)
    if (backLink) {
        backLink.addEventListener('click', function(e) {
            // Cho phép chuyển hướng bình thường, chỉ log để theo dõi
            console.log('Quay lại trang dịch');
        });
    }
});