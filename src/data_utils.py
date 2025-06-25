import numpy as np
import pandas as pd
import nibabel as nib
import cv2
import torch
from torch.utils.data import TensorDataset, DataLoader
import time

def load_and_preprocess_data(config):
    """
    Tải dữ liệu từ các file .nii, tiền xử lý và tạo DataLoader.
    """
    print("Bắt đầu tải và tiền xử lý dữ liệu MRI...")
    start_time = time.time()
    
    # 1. Tải đường dẫn file
    try:
        nii_paths_df = pd.read_csv(config.DATA_PATH_CSV)
        nii_paths = np.array(nii_paths_df)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file CSV tại '{config.DATA_PATH_CSV}'")
        return None

    # 2. Đọc các lát cắt từ file .nii
    img_arrs = []

    # Lấy dữ liệu từ 200 bệnh nhân, bắt đầu từ bệnh nhân thứ 500
    for i in range(200):
        try:
            img = nib.load(nii_paths[i + 500][0])
            data = img.get_fdata()
            # Lấy các lát cắt từ 40 đến 90
            for j in range(40, 90):
                img_arrs.append(data[:, :, j])
        except Exception as e:
            print(f"Cảnh báo: Không thể tải file {nii_paths[i + 500][0]}. Lỗi: {e}")
            continue

    if not img_arrs:
        print("LỖI: Không có ảnh nào được tải. Vui lòng kiểm tra đường dẫn và file.")
        return None

    img_arrs = np.array(img_arrs)
    print(f"Đã tải {img_arrs.shape[0]} lát cắt.")

    # 3. Tiền xử lý ảnh (Windowing, Normalization, Resize)
    num_images = img_arrs.shape[0]
    img_arrs_processed_np = np.empty((num_images, config.IMG_SIZE, config.IMG_SIZE), dtype=np.float32)

    for i in range(num_images):
        img = img_arrs[i]
        p2, p98 = np.percentile(img, (2, 98))
        img_windowed = np.clip(img, p2, p98)
        denominator = (p98 - p2) + 1e-6 # Chống chia cho 0
        img_normalized = (img_windowed - p2) / denominator
        img_resized = cv2.resize(img_normalized, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        img_final = img_resized * 2.0 - 1.0 # Scale về [-1, 1] cho Tanh
        img_arrs_processed_np[i] = img_final

    # Thêm chiều kênh (channel dimension)
    img_arrs_processed_np = np.expand_dims(img_arrs_processed_np, axis=1)
    img_tensor = torch.from_numpy(img_arrs_processed_np)
    
    end_time = time.time()
    print(f"Tiền xử lý hoàn tất trong {end_time - start_time:.2f} giây.")

    # 4. Tạo DataLoader
    print("Đang tạo DataLoader...")
    real_dataset = TensorDataset(img_tensor)
    dataloader = DataLoader(
        real_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    print(f"DataLoader sẵn sàng với {len(dataloader)} batch.")
    
    return dataloader