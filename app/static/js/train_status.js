/**
 * train_status.js
 * Theo dõi tiến trình huấn luyện bằng cách gọi API /train_status
 * Cập nhật thông tin, log và progress bar định kỳ.
 */

// Hàm gọi API để lấy trạng thái huấn luyện
async function fetchTrainStatus() {
    try {
        const response = await fetch('/train_status');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        updateUI(data);
    } catch (error) {
        console.error('Lỗi khi lấy trạng thái huấn luyện:', error);
        document.getElementById('status').innerText = 'Lỗi kết nối';
    }
}

// Cập nhật giao diện với dữ liệu từ server
function updateUI(data) {
    // Cập nhật trạng thái
    const statusSpan = document.getElementById('status');
    if (data.status) {
        statusSpan.innerText = data.status;
    }

    // Epoch hiện tại và tổng số
    const currentEpochSpan = document.getElementById('current-epoch');
    const totalEpochsSpan = document.getElementById('total-epochs');
    if (data.current_epoch !== undefined) {
        currentEpochSpan.innerText = data.current_epoch;
    }
    if (data.total_epochs !== undefined) {
        totalEpochsSpan.innerText = data.total_epochs;
        // Cập nhật progress bar (phần trăm)
        const progress = (data.current_epoch / data.total_epochs) * 100;
        const progressBar = document.getElementById('epoch-progress');
        if (progressBar) {
            progressBar.value = progress;
            progressBar.max = 100;
        }
    }

    // Loss values
    const trainLossSpan = document.getElementById('train-loss');
    const valLossSpan = document.getElementById('val-loss');
    if (data.train_loss !== undefined) {
        trainLossSpan.innerText = data.train_loss.toFixed(6);
    }
    if (data.val_loss !== undefined) {
        valLossSpan.innerText = data.val_loss.toFixed(6);
    }

    // Thời gian epoch
    const epochTimeSpan = document.getElementById('epoch-time');
    if (data.epoch_time !== undefined) {
        epochTimeSpan.innerText = data.epoch_time.toFixed(2);
    }

    // Log chi tiết (nội dung file log)
    const logContent = document.getElementById('log-content');
    if (data.log_content) {
        logContent.innerText = data.log_content;
        // Tự động cuộn xuống cuối
        const logArea = document.querySelector('.log-area');
        if (logArea) logArea.scrollTop = logArea.scrollHeight;
    }
}

// Tải trạng thái lần đầu và thiết lập polling (mỗi 2 giây)
let intervalId = null;
function startPolling() {
    if (intervalId) clearInterval(intervalId);
    fetchTrainStatus(); // gọi ngay
    intervalId = setInterval(fetchTrainStatus, 2000);
}

function stopPolling() {
    if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
    }
}

// Xử lý nút làm mới thủ công
document.addEventListener('DOMContentLoaded', function() {
    startPolling();

    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            fetchTrainStatus();
        });
    }

    // Nút dừng huấn luyện (nếu cần) – yêu cầu API /stop_training
    const stopBtn = document.getElementById('stop-train-btn');
    if (stopBtn) {
        stopBtn.addEventListener('click', async function() {
            try {
                const response = await fetch('/stop_training', { method: 'POST' });
                const result = await response.json();
                alert(result.message || 'Đã yêu cầu dừng');
            } catch (error) {
                console.error(error);
                alert('Không thể dừng huấn luyện');
            }
        });
    }
});

// Dọn dẹp khi rời trang (tuỳ chọn)
window.addEventListener('beforeunload', function() {
    stopPolling();
});