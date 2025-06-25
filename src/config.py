import torch
from pathlib import Path

# --- Cấu hình thiết bị và dữ liệu ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128          # Kích thước ảnh đầu ra
IMG_CHANNELS = 1        # Số kênh ảnh (1 cho ảnh MRI grayscale)
DATA_PATH_CSV = './data/nii_files_path/nii_paths.csv' # Đường dẫn tới file CSV

# --- Cấu hình Training ---
EPOCHS = 600            # Số epochs training
BATCH_SIZE = 32         # Kích thước batch
LR = 0.0001             # Tốc độ học (Adam: 1e-4)
BETA1 = 0.5             # Tham số beta1 của Adam
BETA2 = 0.9             # Tham số beta2 của Adam
CRITIC_ITERATIONS = 4   # Số lần cập nhật Critic cho mỗi lần cập nhật Generator
LAMBDA_GP = 10          # Trọng số của Gradient Penalty

# --- Cấu hình Differential Privacy (DP) ---
ADD_DP_NOISE = False    # Đặt thành True để kích hoạt DP
NOISE_MULTIPLIER = 1.1  # Hệ số nhân nhiễu DP
MAX_GRAD_NORM = 1.0     # Ngưỡng clipping gradient

# --- Cấu hình Lưu trữ & Logging ---
SAMPLE_INTERVAL = 500   # Tần suất lưu ảnh mẫu (tính theo batch)
MODEL_SAVE_INTERVAL = 10 # Tần suất lưu model (tính theo epoch)
OUTPUT_DIR = Path("./output")
FIXED_NOISE_SIZE = 64   # Số ảnh tạo ra để theo dõi tiến trình