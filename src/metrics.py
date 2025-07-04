import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm.notebook import tqdm

def _prepare_images_for_fid(images):
    """
    Chuẩn bị ảnh cho việc tính toán FID.
    - Chuyển đổi từ [-1, 1] sang [0, 1].
    - Chuyển từ 1 kênh (ảnh xám) sang 3 kênh (RGB).
    """
    # Chuyển từ [-1, 1] sang [0, 1]
    images = (images + 1) / 2.0
    
    # Nếu là ảnh 1 kênh, lặp lại 3 lần để thành ảnh 3 kênh
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
        
    return images

def calculate_metrics(generator, dataloader, device, z_dim, num_batches=50):
    """
    Tính toán SSIM và FID cho ảnh được sinh ra so với ảnh thật.
    
    Args:
        generator (torch.nn.Module): Model generator đã được huấn luyện.
        dataloader (DataLoader): DataLoader cho tập dữ liệu thật.
        device (torch.device): Thiết bị để chạy tính toán (CPU hoặc GPU).
        z_dim (int): Số chiều của vector nhiễu z.
        num_batches (int): Số lượng batch sử dụng để đánh giá.
        
    Returns:
        dict: Một dictionary chứa điểm SSIM và FID.
    """
    generator.eval()
    
    # Khởi tạo các chỉ số
    # Đối với SSIM, ảnh có giá trị trong khoảng [-1, 1], nên data_range=2.0
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    
    # FID yêu cầu ảnh 3 kênh, giá trị [0, 255] dạng uint8 hoặc [0, 1] dạng float.
    # Torchmetrics sẽ tự xử lý nếu ta cung cấp ảnh float [0, 1].
    fid_metric = FrechetInceptionDistance(feature=64).to(device)

    with torch.no_grad():
        for i, (real_imgs, _) in tqdm(enumerate(dataloader), total=num_batches, desc="Calculating Metrics"):
            if i >= num_batches:
                break

            real_imgs = real_imgs.to(device)
            
            # Sinh ảnh giả
            noise = torch.randn(real_imgs.size(0), z_dim, 1, 1, device=device)
            fake_imgs = generator(noise)

            # Cập nhật SSIM
            ssim_metric.update(preds=fake_imgs, target=real_imgs)

            # Chuẩn bị ảnh cho FID
            real_fid = _prepare_images_for_fid(real_imgs)
            fake_fid = _prepare_images_for_fid(fake_imgs)

            # Cập nhật FID
            fid_metric.update(imgs=real_fid, real=True)
            fid_metric.update(imgs=fake_fid, real=False)

    metrics = {
        'ssim': ssim_metric.compute().item(),
        'fid': fid_metric.compute().item()
    }
    
    generator.train() # Chuyển generator về lại chế độ huấn luyện
    return metrics
