# Tổng hợp ảnh MRI não với DP-WGAN-GP

Dự án này sử dụng mô hình **Wasserstein GAN with Gradient Penalty (WGAN-GP)** được tăng cường với **Differential Privacy (DP)** để tạo ra các lát cắt ảnh MRI não 2D chân thực và bảo toàn quyền riêng tư của bệnh nhân. Mô hình được huấn luyện bằng PyTorch trên các lát cắt ảnh được trích xuất từ dữ liệu NIfTI 3D.

## Mẫu ảnh được tạo ra

Dưới đây là một số mẫu ảnh MRI não 2D (128x128) được tạo ra bởi mô hình sau khi huấn luyện.

![image](https://github.com/user-attachments/assets/24258dc6-4672-4146-97a3-ef78e867e06b)



## Các tính năng chính

*   **Kiến trúc WGAN-GP**: Sử dụng Wasserstein GAN với Gradient Penalty để giúp quá trình huấn luyện ổn định và tạo ra ảnh chất lượng cao, tránh được hiện tượng sụp đổ mode (mode collapse).
*   **Bảo toàn Riêng tư (Differential Privacy)**: Áp dụng nhiễu Gaussian vào gradient của Discriminator (Critic) trong quá trình huấn luyện (DP-SGD) để cung cấp sự đảm bảo về quyền riêng tư cho dữ liệu huấn luyện.
*   **Xử lý dữ liệu MRI**: Bao gồm các bước tiền xử lý chuyên biệt cho ảnh MRI, chẳng hạn như cửa sổ hóa (windowing) theo phân vị, chuẩn hóa và thay đổi kích thước.
*   **Hỗ trợ NIfTI**: Tải và xử lý trực tiếp các tệp ảnh y tế định dạng `.nii`.
*   **Xây dựng trên PyTorch**: Toàn bộ mô hình và quy trình huấn luyện được xây dựng bằng PyTorch.

## Cấu trúc dự án

```
.
├── mri_dp_wgan.ipynb       # Notebook chính chứa mã nguồn để tiền xử lý, định nghĩa mô hình và huấn luyện
├── data/                     # (Thư mục bạn cần tạo) Nơi chứa dữ liệu
│   └── nii_paths.csv         # File CSV chứa đường dẫn đến các file .nii
│   └── ...                   # Các file .nii hoặc .nii.gz
├── output/                   # (Thư mục được tạo tự động) Chứa kết quả
│   ├── images/               # Chứa các ảnh mẫu được tạo ra trong quá trình huấn luyện
│   └── models/               # Chứa các file checkpoint của mô hình đã lưu
└── README.md                 # File này
```

## Cài đặt và Chuẩn bị

### Yêu cầu

*   Python 3.8+
*   PyTorch 1.10+
*   CUDA (khuyến nghị để huấn luyện trên GPU)

### Hướng dẫn cài đặt

1.  **Clone repository:**
    ```bash
    git clone https://github.com/your-username/mri-synthesis-dp-wgan.git
    cd mri-synthesis-dp-wgan
    ```

2.  **Cài đặt các thư viện cần thiết:**
    Tạo một file `requirements.txt` với nội dung sau:
    ```txt
    numpy
    pandas
    torch
    torchvision
    nibabel
    matplotlib
    opencv-python
    ```
    Sau đó cài đặt bằng pip:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Chuẩn bị dữ liệu:**
    *   Tạo một thư mục `data`.
    *   Đặt tất cả các file ảnh MRI (`.nii` hoặc `.nii.gz`) của bạn vào một nơi có thể truy cập.
    *   Tạo file `data/nii_paths.csv` chứa một cột duy nhất tên là `nii_path` với đường dẫn tuyệt đối hoặc tương đối đến từng file NIfTI.

    Ví dụ nội dung file `nii_paths.csv`:
    ```csv
    nii_path
    /path/to/your/data/subject_001.nii.gz
    /path/to/your/data/subject_002.nii.gz
    ...
    ```

## Cách sử dụng

### 1. Cấu hình

Mở file `mri_dp_wgan.ipynb` và điều chỉnh các tham số trong phần **"Configuration Parameters"** cho phù hợp với nhu cầu của bạn:

*   `LATENT_SIZE`: Kích thước của vector nhiễu đầu vào.
*   `IMG_SIZE`: Kích thước của ảnh đầu ra (ví dụ: 128x128).
*   `EPOCHS`: Số lượng epoch huấn luyện.
*   `BATCH_SIZE`: Kích thước của mỗi batch.
*   `LR`: Tốc độ học (learning rate).
*   `CRITIC_ITERATIONS`: Số lần cập nhật Discriminator cho mỗi lần cập nhật Generator.
*   `LAMBDA_GP`: Hệ số của thành phần gradient penalty.
*   `NOISE_MULTIPLIER`: Hệ số nhiễu cho Differential Privacy. Giá trị càng lớn, độ riêng tư càng cao nhưng có thể ảnh hưởng đến chất lượng ảnh.

### 2. Huấn luyện mô hình

Chạy tất cả các cell trong notebook `mri_dp_wgan.ipynb`. Quá trình huấn luyện sẽ bắt đầu.

*   Các ảnh mẫu sẽ được lưu định kỳ vào thư mục `output/images/`.
*   Các checkpoint của mô hình sẽ được lưu vào `output/models/`.
*   Biểu đồ loss và score sẽ được tạo và lưu lại để theo dõi tiến trình.

### 3. Tạo ảnh từ mô hình đã huấn luyện

Sau khi huấn luyện hoặc nếu bạn có một file checkpoint (`.pt`), bạn có thể sử dụng phần **"Using Pre-trained Weights"** trong notebook:

1.  Cập nhật biến `checkpoint_path` để trỏ đến file `.pt` của bạn.
2.  Chạy các cell trong phần đó để tải trọng số vào Generator.
3.  Chạy cell cuối cùng để tạo ra một batch ảnh mới từ nhiễu ngẫu nhiên.

## Kiến trúc mô hình

*   **Generator**: Sử dụng kiến trúc DCGAN với các lớp `ConvTranspose2d` để tăng độ phân giải (upsampling) từ vector nhiễu đầu vào thành một ảnh 128x128. Lớp kích hoạt cuối cùng là `Tanh` để đưa giá trị pixel về khoảng `[-1, 1]`.
*   **Discriminator (Critic)**: Sử dụng các lớp `Conv2d` để giảm độ phân giải (downsampling) ảnh đầu vào thành một điểm số duy nhất (scalar). Kiến trúc này không sử dụng `BatchNorm` theo khuyến nghị của WGAN-GP và không có hàm kích hoạt ở lớp cuối cùng.
